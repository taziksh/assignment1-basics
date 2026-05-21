import numpy as np
import torch
from einops import rearrange
import wandb
from datetime import datetime
import os
import json
from dataclasses import asdict
import tyro
from cs336_basics.trainer import (
    get_batch,
    get_lr_cosine_schedule,
    cross_entropy,
    save_checkpoint,
    gradient_clipping,
    AdamWOptim,
)
from cs336_basics.transformer import TransformerLM
from cs336_basics.config import TrainingConfig


def train(cfg):
    if cfg.wandb:
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))
    prefix = f"{wandb.run.name}_" if cfg.wandb else ""

    run_dir = f"runs/{prefix}train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_data = np.load(cfg.train_data, mmap_mode="r")
    val_data = np.load(cfg.val_data, mmap_mode="r")

    model = TransformerLM(
        cfg.model.vocab_size, cfg.model.context_length, cfg.model.d_model, cfg.model.num_layers, cfg.model.num_heads, cfg.model.d_ff, cfg.model.rope_theta
    ).to(cfg.device)
    model = torch.compile(model, backend="aot_eager")

    optim = AdamWOptim(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=cfg.optim.eps, betas=[cfg.optim.beta_1, cfg.optim.beta_2]
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
    for i in range(total_steps):
        x, y = get_batch(train_data, batch_size, context_length, device=cfg.device)
        y = rearrange(y, "b s -> (b s)")
        logits = model(x)
        logits = rearrange(logits, "b s v -> (b s) v")

        loss = cross_entropy(logits, y)

        optim.zero_grad()
        loss.backward()
        # Gradient clip at 1.0 following example of GPT-3, LlaMA, PaLM
        gradient_clipping(model.parameters(), 1.0)

        lr = get_lr_cosine_schedule(
            it=i,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=total_steps,
        )
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()

        if i % cfg.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch(val_data, batch_size, context_length, device=cfg.device)
                val_logits = model(val_x)
                val_logits = rearrange(val_logits, "b s v -> (b s) v")
                val_y = rearrange(val_y, "b s -> (b s)")

                val_loss = cross_entropy(val_logits, val_y)

                if cfg.wandb:
                    wandb.log({"val/loss": val_loss.item()}, step=i)

            model.train()

        if i > 0 and (i % cfg.checkpoint_interval == 0 or i == total_steps - 1):
            save_checkpoint(model, optim, i, f"{run_dir}/ckpt_step_{i}.pt")

        if cfg.wandb and i % cfg.log_interval == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    # "lr": lr,
                },
                step=i,
            )


if __name__ == "__main__":
    cfg = tyro.cli(TrainingConfig)

    has_steps = cfg.total_steps is not None
    has_tokens = cfg.total_tokens is not None

    if has_steps == has_tokens:
        raise ValueError("Provide either --total-steps or --total-tokens")

    if has_tokens:
        cfg.total_steps = cfg.total_tokens // (cfg.batch_size * cfg.model.context_length)

    train(cfg)
