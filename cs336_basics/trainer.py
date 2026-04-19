from jaxtyping import Float, Int
from collections.abc import Callable
from typing import Optional, Iterable, IO, BinaryIO, Any
import torch
import math
import numpy.typing as npt
import os


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    batch_size = targets.shape[-1]
    max_logit = torch.max(inputs, dim=-1, keepdim=True).values
    return torch.mean(
        max_logit.squeeze(dim=-1)
        + torch.log(torch.sum(torch.exp(inputs - max_logit), dim=-1))
        - inputs[torch.arange(batch_size), targets]
    )


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

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
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
                p.data -= lr * weight_decay * p.data
                state["m"] = beta_1 * state["m"] + (1 - beta_1) * p.grad
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * p.grad**2
                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)
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
):
    result = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(result, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    result = torch.load(src)
    model.load_state_dict(result["model"])
    optimizer.load_state_dict(result["optimizer"])
    return result["iteration"]
