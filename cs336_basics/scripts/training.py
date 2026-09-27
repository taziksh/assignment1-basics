import numpy as np
import torch
from einops import rearrange
import wandb
from datetime import datetime
import os
import json
import time
from dataclasses import asdict
import tyro
from cs336_basics.trainer import (
    get_batch,
    get_lr_cosine_schedule,
    cross_entropy,
    save_checkpoint,
    gradient_clipping,
    AdamWOptim,
    MuonOptim
)
from cs336_basics.transformer import TransformerLM
from cs336_basics.config import TrainingConfig


@torch.compile(backend="inductor" if torch.cuda.is_available() else "aot_eager")
def fwd_loss(x, y, model, device):
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(x)
        logits = rearrange(logits, "b s v -> (b s) v")
        loss = cross_entropy(logits.float(), y)
    return loss


def train(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.set_float32_matmul_precision("high")
    if cfg.wandb:
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        wandb.define_metric("wallclock_min")
        wandb.define_metric("train/*", step_metric="wallclock_min")
        wandb.define_metric("val/*", step_metric="wallclock_min")
    prefix = f"{wandb.run.name}_" if cfg.wandb else ""

    run_dir = f"runs/{prefix}train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_data = np.load(cfg.train_data, mmap_mode="r")
    val_data = np.load(cfg.val_data, mmap_mode="r")

    model = TransformerLM(
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.d_model,
        cfg.model.num_layers,
        cfg.model.num_heads,
        cfg.model.d_ff,
        cfg.model.rope_theta,
    ).to(cfg.device)

    adam_parameters = []
    muon_parameters = []

    for name, param in model.named_parameters():
        # 2D parameters only
        if param.ndim == 2 and name.startswith("layers."):
            muon_parameters.append(param)
        else:
            adam_parameters.append(param)

    model = torch.compile(model) if cfg.device == "cuda" else torch.compile(model, backend="aot_eager")

    optim = AdamWOptim(
        adam_parameters,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
        betas=[cfg.optim.beta_1, cfg.optim.beta_2],
    )

    optim2 = MuonOptim(
        muon_parameters,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )

    # n=1 batch to test overfitting
    # x, y = get_batch(train_data, cfg.batch_size, cfg.model.context_length, device=cfg.device)
    # y = rearrange(y, "b s -> (b s)")

    batch_size = cfg.batch_size
    context_length = cfg.model.context_length

    min_lr = 0.1 * cfg.optim.lr
    max_lr = cfg.optim.lr
    total_steps = cfg.total_steps
    warmup_steps = int(0.05 * total_steps)
    start_time = time.perf_counter()
    last_log_minutes = 0.0
    window_minutes = 0.0
    for i in range(total_steps):
        x, y = get_batch(train_data, batch_size, context_length, device=cfg.device)
        y = rearrange(y, "b s -> (b s)")
        loss = fwd_loss(x, y, model, cfg.device)

        optim.zero_grad()
        optim2.zero_grad()
        loss.backward()
        # Gradient clip at 1.0 following example of GPT-3, LlaMA, PaLM
        gradient_clipping(adam_parameters, 1.0)

        lr = get_lr_cosine_schedule(
            it=i,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=total_steps,
        )
        for group in optim.param_groups:
            group["lr"] = lr
        for group in optim2.param_groups:
            group["lr"] = lr
        optim.step()
        optim2.step()

        elapsed_minutes = (time.perf_counter() - start_time) / 60
        is_over_budget = cfg.max_minutes is not None and elapsed_minutes + window_minutes >= cfg.max_minutes

        if i % cfg.val_interval == 0:
            model.eval()
            with (
                torch.no_grad(),
                torch.autocast(device_type=cfg.device, dtype=torch.bfloat16, enabled=cfg.device == "cuda"),
            ):
                val_x, val_y = get_batch(val_data, batch_size, context_length, device=cfg.device)
                val_logits = model(val_x)
                val_logits = rearrange(val_logits, "b s v -> (b s) v")
                val_y = rearrange(val_y, "b s -> (b s)")

                val_loss = cross_entropy(val_logits.float(), val_y)

                if cfg.wandb:
                    wandb.log(
                        {"val/loss": val_loss.item(), "wallclock_min": elapsed_minutes},
                        step=i,
                    )

            model.train()

        if i > 0 and (i % cfg.checkpoint_interval == 0 or i == total_steps - 1 or is_over_budget):
            save_checkpoint(model, optim, i, f"{run_dir}/ckpt_step_{i}.pt", optim2)

        if i > 0 and i % cfg.log_interval == 0:
            window_minutes = elapsed_minutes - last_log_minutes
            last_log_minutes = elapsed_minutes

            loss_val = loss.item()
            print(f"[train] step {i} loss {loss_val:.4f} {elapsed_minutes:.3f}m")
            if cfg.wandb:
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "wallclock_min": elapsed_minutes,
                        # "lr": lr,
                    },
                    step=i,
                )
        if is_over_budget:
            break

    model.eval()
    final_val_loss = 0.0
    with torch.no_grad(), torch.autocast(device_type=cfg.device, dtype=torch.bfloat16, enabled=cfg.device == "cuda"):
        for _ in range(cfg.final_val_batches):
            val_x, val_y = get_batch(val_data, batch_size, context_length, device=cfg.device)
            val_logits = model(val_x)
            val_logits = rearrange(val_logits, "b s v -> (b s) v")
            val_y = rearrange(val_y, "b s -> (b s)")

            final_val_loss += cross_entropy(val_logits.float(), val_y).item()
    final_val_loss /= cfg.final_val_batches

    print(f"[eval] final val loss {final_val_loss:.4f} over {cfg.final_val_batches} batches")
    if cfg.wandb:
        wandb.run.summary["val/final_loss"] = final_val_loss


if __name__ == "__main__":
    cfg = tyro.cli(TrainingConfig)

    has_steps = cfg.total_steps is not None
    has_tokens = cfg.total_tokens is not None

    if has_steps == has_tokens:
        raise ValueError("Provide either --total-steps or --total-tokens")

    if has_tokens:
        cfg.total_steps = cfg.total_tokens // (cfg.batch_size * cfg.model.context_length)

    train(cfg)
