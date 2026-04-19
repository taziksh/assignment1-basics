import argparse
import torch

from cs336_basics.scripts.training import model_parser, optim_parser
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.trainer import load_checkpoint, AdamWOptim
from cs336_basics.transformer import TransformerLM, softmax


def tokenizer_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("tokenizer")
    # TODO: make these non hardcoded
    g.add_argument(
        "--vocab-filepath", type=str, default="runs/TinyStoriesV2-GPT4-train_vocab10000_20260413_180136/vocab.pkl"
    )
    g.add_argument(
        "--merges-filepath", type=str, default="runs/TinyStoriesV2-GPT4-train_vocab10000_20260413_180136/merges.pkl"
    )
    g.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"])
    g.add_argument("--max-tokens", type=int, default=2048)
    return p


def main_parser():
    p = argparse.ArgumentParser(parents=[model_parser(), optim_parser(), tokenizer_parser()])
    p.add_argument("--prompt", type=str, required=True)
    # TODO: make these non hardcoded
    p.add_argument("--checkpoint", type=str, default="runs/train_20260418_221619/ckpt_step_9900.pt")
    p.add_argument("--device", type=str, choices=["mps", "cuda", "cpu"], default="mps")
    return p


if __name__ == "__main__":
    args = main_parser().parse_args()

    prompt = args.prompt
    ckpt = args.checkpoint
    device = args.device

    vocab = args.vocab_filepath
    merges = args.merges_filepath
    special_tokens = args.special_tokens
    max_tokens = args.max_tokens

    ctx_len = args.context_length

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta
    ).to(args.device)

    optim = AdamWOptim(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=[args.beta_1, args.beta_2]
    )

    load_checkpoint(ckpt, model, optim)

    tokenizer = Tokenizer.from_files(vocab, merges, special_tokens)
    # TODO this assumes specifically "<|endoftext|>""
    eot_token_id = tokenizer.encode(special_tokens[0])[0]

    tokens = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long)
    tokens = tokens.unsqueeze(0)
    count = 0

    while count < max_tokens:
        logits = model(tokens)
        next_token_logits = logits[:, -1, :]
        sm = softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(sm, 1)
        count += 1
        if next_token == eot_token_id:
            break
        tokens = torch.concat([tokens, next_token], dim=-1)
        tokens = tokens[:, -ctx_len:]

    print(tokenizer.decode(tokens.squeeze(0).tolist()))
