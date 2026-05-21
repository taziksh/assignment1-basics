import torch
import tyro

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.trainer import load_checkpoint, AdamWOptim
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.config import DecodingConfig


if __name__ == "__main__":
    cfg = tyro.cli(DecodingConfig)

    prompt = cfg.prompt
    ckpt = cfg.checkpoint
    device = cfg.device

    assert cfg.temperature > 0
    temperature = cfg.temperature
    p = cfg.top_p

    vocab = cfg.vocab_filepath
    merges = cfg.merges_filepath
    special_tokens = cfg.special_tokens
    max_tokens = cfg.max_tokens

    ctx_len = cfg.model.context_length

    model = TransformerLM(
        cfg.model.vocab_size, cfg.model.context_length, cfg.model.d_model, cfg.model.num_layers, cfg.model.num_heads, cfg.model.d_ff, cfg.model.rope_theta
    ).to(cfg.device)

    optim = AdamWOptim(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=cfg.optim.eps, betas=[cfg.optim.beta_1, cfg.optim.beta_2]
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
        next_token_logits = next_token_logits / temperature
        probs = softmax(next_token_logits, dim=-1)
        if p:
            sorted_vals, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=-1)
            mask = (cumsum - sorted_vals) < p
            sorted_vals *= mask
            sample = torch.multinomial(sorted_vals, 1)
            next_token = sorted_indices.gather(dim=-1, index=sample)
        else:
            next_token = torch.multinomial(probs, 1)
        count += 1
        if next_token == eot_token_id:
            break
        tokens = torch.concat([tokens, next_token], dim=-1)
        tokens = tokens[:, -ctx_len:]

    print(tokenizer.decode(tokens.squeeze(0).tolist()))
