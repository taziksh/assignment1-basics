import regex as re, collections, os, time, pickle
import argparse
from typing import BinaryIO
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import json
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_regex = ""
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        if special_tokens:
            self.special_tokens.sort(key=len, reverse=True)
            self.special_token_regex = "|".join(re.escape(t) for t in self.special_tokens)   
        self.merge_order = {pair: i for i, pair in enumerate(self.merges)}     

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ) -> None:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, pt: list[bytes]) -> list[bytes]:
        while True:
            candidate_pairs = [pair for pair in zip(pt, pt[1:]) if pair in self.merge_order]
            if not candidate_pairs: 
                break

            min_pair = min(candidate_pairs, key=self.merge_order.get)
            new_pt = []
            i = 0

            while i < len(pt):
                if pt[i] == min_pair[0] and i < len(pt)-1 and pt[i+1] == min_pair[1]:
                    new_pt.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_pt.append(pt[i])
                    i += 1
            pt = new_pt
        return pt

    def encode(self, text: str) -> list[int]:
        token_ids = []
        if self.special_token_regex:
            text_chunks = re.split(f"({self.special_token_regex})", text)
        else:
            text_chunks = [text]
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.reverse_vocab[chunk.encode('utf-8')])
                continue
            tokens = re.finditer(PAT, chunk)
            pre_tokens = [[bytes([byte]) for byte in token.group().encode('utf-8')] for token in tokens]
            for pt in pre_tokens:
                merged = self._apply_merges(pt)
                token_ids.extend(self.reverse_vocab[b] for b in merged)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        byte_array = []
        for id in ids:
            byte_array.append(self.vocab[id])

        return b"".join(byte_array).decode("utf-8", errors="replace")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge_vocab(pair, vocab, pairs):
    new_vocab = {}
    bigram = pair[0] + pair[1]

    for word in vocab:
        new_word = []
        i = 0
        freq = vocab[word]
        while i < len(word):
            if word[i] == pair[0] and i < len(word)-1 and word[i+1] == pair[1]:
                new_word.append(bigram)

                if i > 0:
                    pairs[word[i-1], bigram] += freq
                    pairs[word[i-1], word[i]] -= freq

                if i < len(word)-2:
                    pairs[bigram, word[i+2]] += freq
                    pairs[word[i+1], word[i+2]] -= freq

                pairs[word[i], word[i+1]] -= freq
                
                i += 2

            else:
                new_word.append(word[i])
                i += 1

        new_vocab[tuple(new_word)] = freq
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
        byte_tokens = [tuple(bytes([byte]) for byte in token.group().encode('utf-8')) for token in tokens]        

        for key in byte_tokens:
            local_vocab[key] += 1
    return local_vocab

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    num_processes = kwargs.get("num_processes", 8)
    num_merges = vocab_size - 256 - len(special_tokens)

    with open(input_path, "rb") as f:
        # TODO: note that the byte-encoded special token is hardcoded
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

    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pairs[word[i], word[i+1]] += freq

    for _ in range(num_merges):
        best = max(pairs, key=lambda pair: (pairs.get(pair), pair))
        vocab = merge_vocab(best, vocab, pairs)
        merges.append(best)
        pairs = collections.defaultdict(int, {k: v for k, v in pairs.items() if v > 0})
        
    final_vocab = {}
    next_id = 0

    for t in special_tokens:
        final_vocab[next_id] = t.encode("utf-8")
        next_id += 1

    for i in range(256):
        final_vocab[next_id] = bytes([i])
        next_id += 1

    for pair in merges:
        final_vocab[next_id] = pair[0] + pair[1]
        next_id += 1

    return final_vocab, merges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()
    num_processes = args.num_processes

    # input_path = "data/bpe_example.txt"
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"

    input_path = "data/owt_valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 32000

    start = time.time()
    final_vocab, merges = train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens, num_processes=num_processes)
    duration = time.time() - start
    print(f"{duration:.2f}s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{Path(input_path).stem}_vocab{vocab_size}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(final_vocab, f)
    with open(os.path.join(run_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    longest = max(final_vocab.values(), key=len)    


    config = {
        "input_path": str(input_path),
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "num_processes": num_processes,
        "duration_seconds": duration,
        "longest_token": longest.decode("utf-8", errors="replace"),
        "longest_token_length": len(longest)
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)