import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs 

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {
    'l o w': 5,
    'l o w e r': 2,
    'w i d e s t': 3,
    'n e w e s t': 6
}

num_merges = 6

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=lambda pair: (pairs.get(pair), pair))
    vocab = merge_vocab(best, vocab)
    print(vocab)
