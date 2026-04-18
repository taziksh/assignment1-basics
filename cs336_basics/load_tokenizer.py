import os
import pickle


def load_tokenizer(run_dir):
    with open(os.path.join(run_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(run_dir, "merges.pkl"), "rb") as f:
        merges = pickle.load(f)
    return vocab, merges


if __name__ == "__main__":
    run_dir = "runs/TinyStoriesV2-GPT4-train_vocab1000_20260413_172529"
    vocab, merges = load_tokenizer(run_dir)
    breakpoint()
    longest = max(vocab.values(), key=len)
    print(f"Longest token: {longest!r} ({len(longest)} bytes)")
