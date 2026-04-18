from cs336_basics.tokenizer import Tokenizer
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import json

if __name__ == "__main__":
    tokenizer_dir = "runs/TinyStoriesV2-GPT4-train_vocab10000_20260413_180136"
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    tokenizer = Tokenizer.from_files(
        vocab_filepath=f"{tokenizer_dir}/vocab.pkl",
        merges_filepath=f"{tokenizer_dir}/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"runs/encoded_{Path(input_path).stem}_{Path(tokenizer_dir).name}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        gen = tokenizer.encode_iterable(f)
        arr = np.fromiter(gen, dtype=np.uint16)
        np.save(os.path.join(out_dir, "tokens.npy"), arr)

    config = {
        "input_path": input_path,
        "tokenizer_dir": tokenizer_dir,
        "num_tokens": int(arr.shape[0]),
        "max_id": int(arr.max()),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
