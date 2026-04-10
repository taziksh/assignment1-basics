import regex as re, collections, os

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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    num_merges = vocab_size - 256 - len(special_tokens)

    special_token_regex = "|".join(re.escape(t) for t in special_tokens)
    vocab = collections.defaultdict(int)

    with open(input_path) as f:
        text = f.read()
        docs = re.split(special_token_regex, text)

        for doc in docs:
            tokens = re.finditer(PAT, doc)
            byte_tokens = [tuple(bytes([b]) for b in token.group().encode('utf-8')) for token in tokens]        

            for key in byte_tokens:
                vocab[key] += 1

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
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    input_path = "data/bpe_example.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 260

    train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
