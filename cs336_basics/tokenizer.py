import regex as re, collections, os, sys, time
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pairs[word[i], word[i+1]] += freq
    return pairs 

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = pair[0] + pair[1]
    for word in vocab:
        new_word = []
        i = 0
        while i < len(word):
            if word[i] == pair[0] and i < len(word)-1 and word[i+1] == pair[1]:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = vocab[word]
    return new_vocab

def process_chunk(args):
    start, end, input_path, special_tokens = args
    special_token_regex = "|".join(re.escape(t) for t in special_tokens)
    local_vocab = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    docs = re.split(special_token_regex, chunk)
    for doc in docs:
        tokens = re.finditer(PAT, doc)
        byte_tokens = [tuple(bytes([b]) for b in token.group().encode('utf-8')) for token in tokens]        

        for key in byte_tokens:
            local_vocab[key] += 1
    return local_vocab

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    num_processes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_merges = vocab_size - 256 - len(special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    args_list = [(s, e, input_path, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    if num_processes == 1:
        results = [process_chunk(args) for args in args_list]
    else:
        with Pool(num_processes) as pool:
            results = pool.map(process_chunk, args_list)

    vocab = collections.defaultdict(int)
    for local_vocab in results:
        for key, count in local_vocab.items():
            vocab[key] += count

    assert vocab_size > 256 + len(special_tokens)

    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=lambda pair: (pairs.get(pair), pair))
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    print(vocab)
    print(pairs)

if __name__ == "__main__":
    # input_path = "data/bpe_example.txt"
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 260

    start = time.time()
    train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    print(f"{time.time() - start:.2f}s")