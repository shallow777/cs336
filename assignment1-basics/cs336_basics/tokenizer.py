import os
import regex
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    pretoken_freq = defaultdict(int)

    if special_tokens:
        special_pattern = "|".join(regex.escape(tok) for tok in special_tokens)
        parts = regex.split(f"({special_pattern})", text)
        special_set = set(special_tokens)
    else:
        parts = [text]
        special_set = set()

    for part in parts:
        if not part:
            continue
        if part in special_set:
            continue

        for match in regex.finditer(PAT, part):
            token_bytes = match.group(0).encode("utf-8")
            if len(token_bytes) == 0:
                continue
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            pretoken_freq[token_tuple] += 1

    def get_pair_freq(pretoken_freq):
        pair_freq = defaultdict(int)
        for token, count in pretoken_freq.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_freq[pair] += count
        return pair_freq

    pair_freq = get_pair_freq(pretoken_freq)

    vocab = {}
    next_id = 0

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    merges = []
    num_merges = vocab_size - len(vocab)

    if num_merges <= 0:
        return vocab, merges

    def merge_token(token, best_pair, new_symbol):
        merged = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                merged.append(new_symbol)
                i += 2
            else:
                merged.append(token[i])
                i += 1
        return tuple(merged)

    for _ in range(num_merges):
        if not pair_freq:
            break

        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        new_symbol = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        vocab[next_id] = new_symbol
        next_id += 1

        new_pretoken_freq = defaultdict(int)

        for token, count in pretoken_freq.items():
            if len(token) < 2:
                new_pretoken_freq[token] += count
                continue

            new_token = merge_token(token, best_pair, new_symbol)
            new_pretoken_freq[new_token] += count

        pretoken_freq = new_pretoken_freq
        pair_freq = get_pair_freq(pretoken_freq)

    return vocab, merges
