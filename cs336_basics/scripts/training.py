import argparse
import numpy as np
import torch
from einops import rearrange
import wandb
from datetime import datetime
import os
import json
from cs336_basics.trainer import get_batch, get_lr_cosine_schedule, cross_entropy, save_checkpoint, gradient_clipping, AdamWOptim
from cs336_basics.transformer import TransformerLM
from cs336_basics.scripts.cli import model_parser, optim_parser


def data_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("data")
    g.add_argument("--train-data", type=str, required=True)
    g.add_argument("--val-data", type=str, required=True)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--total-steps", type=int, default=10000)
    return p


def logging_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("logging")
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb-project", type=str, default="cs336-1")
    g.add_argument("--log-interval", type=int, default=100)
    return p


def main_parser():
    p = argparse.ArgumentParser(parents=[model_parser(), optim_parser(), data_parser(), logging_parser()])
    p.add_argument("--device", type=str, choices=["mps", "cuda", "cpu"], default="mps")
    p.add_argument("--checkpoint-interval", type=int, default=10000)
    p.add_argument("--val-interval", type=int, default=10)
    return p

def train(args):
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        for k, v in dict(wandb.config).items():
            setattr(args, k, v)
    prefix = f"{wandb.run.name}_" if args.wandb else ""

    run_dir = f"runs/{prefix}train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(args.device)
    model = torch.compile(model, backend="aot_eager")

    optim = AdamWOptim(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=[args.beta_1, args.beta_2]
    )

    # n=1 batch to test overfitting
    # x, y = get_batch(train_data, args.batch_size, args.context_length, device=args.device)
    # y = rearrange(y, "b s -> (b s)")


    min_lr = 0.1 * args.lr
    max_lr = args.lr
    total_steps = args.total_steps
    warmup_steps = int(0.05 * args.total_steps)
    for i in range(args.total_steps):
        x, y = get_batch(train_data, args.batch_size, args.context_length, device=args.device)
        y = rearrange(y, "b s -> (b s)")
        logits = model(x)
        logits = rearrange(logits, "b s v -> (b s) v")

        loss = cross_entropy(logits, y)

        optim.zero_grad()
        loss.backward()
        # Gradient clip at 1.0 following example of GPT-3, LlaMA, PaLM
        gradient_clipping(model.parameters(), 1.0)

        lr = get_lr_cosine_schedule(it=i, max_learning_rate=max_lr, min_learning_rate=min_lr, warmup_iters=warmup_steps, cosine_cycle_iters=total_steps)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()

        if i % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, device=args.device)
                val_logits = model(val_x)
                val_logits = rearrange(val_logits, "b s v -> (b s) v")
                val_y = rearrange(val_y, "b s -> (b s)")

                val_loss = cross_entropy(val_logits, val_y)

                if args.wandb:
                    wandb.log({"val/loss": val_loss.item()}, step=i)

            model.train()

        if i > 0 and i % args.checkpoint_interval == 0:
            save_checkpoint(model, optim, i, f"{run_dir}/ckpt_step_{i}.pt")

        if args.wandb and i % args.log_interval == 0:
            wandb.log(
                {"train/loss": loss.item(),
                "lr": lr,
                }, step=i)

if __name__ == "__main__":
    args = main_parser().parse_args()
    train(args)