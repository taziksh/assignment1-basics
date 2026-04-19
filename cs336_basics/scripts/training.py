import argparse
import numpy as np
import torch
from einops import rearrange
import wandb
from datetime import datetime
import os
from cs336_basics.trainer import get_batch, cross_entropy, save_checkpoint, AdamWOptim
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
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--val-interval", type=int, default=10)
    return p


if __name__ == "__main__":
    args = main_parser().parse_args()

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(args.device)

    optim = AdamWOptim(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=[args.beta_1, args.beta_2]
    )

    run_dir = f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # n=1 batch to test overfitting
    # x, y = get_batch(train_data, args.batch_size, args.context_length, device=args.device)
    # y = rearrange(y, "b s -> (b s)")
    for i, step in enumerate(range(args.total_steps)):
        x, y = get_batch(train_data, args.batch_size, args.context_length, device=args.device)
        y = rearrange(y, "b s -> (b s)")
        logits = model(x)
        logits = rearrange(logits, "b s v -> (b s) v")

        loss = cross_entropy(logits, y)

        optim.zero_grad()
        loss.backward()
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
            wandb.log({"train/loss": loss.item()}, step=i)
