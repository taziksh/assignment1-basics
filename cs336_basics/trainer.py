from jaxtyping import Float, Int
from collections.abc import Callable
from typing import IO, BinaryIO
from collections.abc import Iterable
import torch
import math
import numpy.typing as npt
import os


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    return torch.mean(torch.logsumexp(inputs, dim=-1) - torch.gather(inputs, -1, targets.unsqueeze(1)).squeeze(1))

# https://docs.modula.systems/algorithms/newton-schulz/
@torch.compile(dynamic=False, backend="inductor" if torch.cuda.is_available() else "aot_eager")
def newton_schulz_ortho(M: torch.Tensor, k: int):
    # quintic equation
    a = 3.4445
    b = -4.7750
    c = 2.0315
    
    eps = 1e-7

    if M.shape[0] > M.shape[1]:
        return newton_schulz_ortho(M.T, k).T

    in_dtype = M.dtype
    M = M.to(torch.bfloat16)

    M_F = torch.linalg.norm(M, ord='fro')
    X = M / (M_F + eps)
    for _ in range(k):
        gram = X@X.T
        gram_times_X = gram @ X
        X = a * X + b * gram_times_X + c * gram @ gram_times_X

    return X.to(in_dtype)


class MuonOptim(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, weight_decay):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        super().__init__(params, defaults)

    # https://arxiv.org/html/2502.16982v1#S2.SS1
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not state:
                    state["B"] = torch.zeros_like(p)

                state["B"] = momentum * state["B"] + p.grad
                ortho_update = newton_schulz_ortho(momentum * state["B"] + p.grad, 5)
                # Following Moonshot, match Muon's update RMS to Adam
                with torch.no_grad():
                    p -= lr * (0.2 * math.sqrt(max(p.shape)) * ortho_update + weight_decay * p)

        return loss


# TODO: move to new file, optim.py
# Hyperparameter defaults are from AdamW paper,
# except for weight_decay, which is from GPT-3/LLaMA
class AdamWOptim(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), eps=10e-6, weight_decay=0.1, lr=0.001):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 1

                alpha_t = lr * math.sqrt(1 - beta_2 ** state["t"]) / (1 - beta_1 ** state["t"])
                with torch.no_grad():
                    p -= lr * weight_decay * p
                state["m"] = beta_1 * state["m"] + (1 - beta_1) * p.grad
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * p.grad**2
                with torch.no_grad():
                    p -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] += 1

        return loss


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    curr_lr = None
    if it < warmup_iters:
        curr_lr = it * max_learning_rate / warmup_iters
    elif it > cosine_cycle_iters:
        curr_lr = min_learning_rate
    else:
        curr_lr = min_learning_rate + (1 / 2) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    return curr_lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 10e-6
    total_norm_sq = 0
    for p in parameters:
        if p.grad is None:
            continue
        total_norm_sq += torch.sum(p.grad**2)

    total_norm = torch.sqrt(total_norm_sq)
    if total_norm < max_l2_norm:
        return

    for p in parameters:
        if p.grad is None:
            continue
        p.grad *= (max_l2_norm) / (total_norm + eps)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(low=0, high=len(dataset) - context_length, size=(batch_size, 1))
    offsets = torch.arange(start=0, end=context_length).unsqueeze(dim=0)
    inputs = starts + offsets
    outputs = inputs + 1
    return torch.from_numpy(dataset[inputs.to("cpu")]).int().to(device), torch.from_numpy(
        dataset[outputs.to("cpu")]
    ).int().to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    optimizer2: torch.optim.Optimizer | None = None,
):
    result = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    if optimizer2 is not None:
        result["optimizer2"] = optimizer2.state_dict()
    torch.save(result, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer, optimizer2: torch.optim.Optimizer | None = None
) -> int:
    result = torch.load(src)
    model.load_state_dict(result["model"])
    optimizer.load_state_dict(result["optimizer"])
    if optimizer2 is not None:
        optimizer2.load_state_dict(result["optimizer2"])
    return result["iteration"]
