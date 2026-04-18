import argparse
import numpy as np
from einops import rearrange
import wandb
from datetime import datetime
import os
from cs336_basics.trainer import get_batch, cross_entropy, save_checkpoint, AdamWOptim
from cs336_basics.transformer import TransformerLM


def model_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("model")
    g.add_argument("--vocab-size", type=int, default=10000)
    g.add_argument("--context-length", type=int, default=256)
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument("--num-heads", type=int, default=16)
    g.add_argument("--d-ff", type=int, default=1344)
    g.add_argument("--rope-theta", type=float, default=10000)
    return p


def optim_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("optim")
    g.add_argument("--lr", type=float, default=0.001)
    g.add_argument("--weight-decay", type=float, default=0)
    g.add_argument("--beta-1", type=float, default=0.9)
    g.add_argument("--beta-2", type=float, default=0.999)
    g.add_argument("--eps", type=float, default=10e-6)
    return p


def data_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("data")
    g.add_argument("--train-data", type=str)  # , required=True)
    g.add_argument("--val-data", type=str)  # , required=True)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--total_steps", type=int, default=1000)
    return p


def logging_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("logging")
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb-project", type=str, default="cs336-1")
    g.add_argument("--log-interval", type=int, default=10)
    return p


def main_parser():
    p = argparse.ArgumentParser(parents=[model_parser(), optim_parser(), data_parser(), logging_parser()])
    p.add_argument("--device", type=str, choices=["mps", "cuda", "cpu"], default="mps")
    p.add_argument("--checkpoint_interval", type=int, default=100)
    return p


if __name__ == "__main__":
    args = main_parser().parse_args()

    # if args.train_data:
    train_data = np.load(args.train_data, mmap_mode="r")

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

    for i, step in enumerate(range(args.total_steps)):
        x, y = get_batch(train_data, args.batch_size, args.context_length, device=args.device)
        logits = model(x)
        logits = rearrange(logits, "b s v -> (b s) v")
        y = rearrange(y, "b s -> (b s)")

        loss = cross_entropy(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i > 0 and i % args.checkpoint_interval == 0:
            save_checkpoint(model, optim, i, f"{run_dir}/ckpt_step_{i}.pt")

        if args.wandb and i % args.log_interval == 0:
            wandb.log({"train/loss": loss.item()}, step=i)
        print(f"Loss: {loss.item()}")
