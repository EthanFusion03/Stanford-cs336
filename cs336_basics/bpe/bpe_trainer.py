import regex as re
import os
import pickle
import time
import multiprocessing
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import BinaryIO

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: list[tuple[bytes, bytes]]  # index1,index2 -> new_index

class Node:
    '''A node class for doubly linked list.'''
    def __init__(self, value, prev=None, next=None):
        self.value = value
        self.prev = prev
        self.next = next

def get_pair_stats(word_freqs):
    counts = defaultdict(int)
    for word_head, freq in word_freqs.items():
        node = word_head
        while node and node.next:
            counts[(node.value, node.next.value)] += freq
            node = node.next
    return counts

def gpt2_pretokenization(text: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, text)

# --- Worker Function for Multiprocessing ---
def worker_process_chunk(input_path: str, start_byte: int, end_byte: int, special_tokens: list[str]) -> Counter:
    """
    Reads a file chunk, splits it by special tokens, and then counts
    pre-token frequencies in each resulting sub-chunk.
    """
    local_word_counts = Counter()
    # Special token as hard boundaries
    # special_tokens = ["<|endoftext|>"]
    escaped_tokens = [re.escape(s) for s in special_tokens]
    delimiter_pattern = '|'.join(escaped_tokens)

    with open(input_path, 'rb') as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
    
    chunk_text = chunk_bytes.decode(encoding='utf-8', errors='ignore')

    # Split the chunk into smaller pieces using the special tokens as delimiters.
    sub_chunks = re.split(f"({delimiter_pattern})", chunk_text)

    for sub_chunk in sub_chunks:
        if sub_chunk in special_tokens or not sub_chunk:
            continue
        pre_tokenizer = gpt2_pretokenization(sub_chunk)
        for match in pre_tokenizer:
            local_word_counts[match.group(0)] += 1
    return local_word_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    """
    Optimized BPE training using a doubly linked list and a cached counts dictionary.
    """
    assert vocab_size >= 256 + len(special_tokens), "vocab_size must be large enough to hold the base vocabulary and special tokens."
    num_merges = max(vocab_size - 256 - len(special_tokens), 0)

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}    # index -> bytes
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode('utf-8')

    # --- Step 1: Parallel pre-tokenization ---
    num_processes = os.cpu_count() or 1
    print("Starting parallel pre-tokenization...")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(num_processes) as pool:
        list_of_counts = pool.starmap(worker_process_chunk, tqdm(tasks, desc="Pretokenizing Chunks"))

    print("Aggregating results...")
    per_word_counts = Counter()
    for local_counts in list_of_counts:
        per_word_counts.update(local_counts)
    
    # --- Step 2: Convert words to Doubly Linked Lists ---
    # word_freqs now maps the *head node* of a linked list to the word's frequency.
    print("Pre-tokenization complete. Starting BPE merges...")

    word_freqs = {}
    for word, freq in per_word_counts.items():
        byte_values = word.encode(encoding='utf-8')
        # Add a guard for empty pre-tokens
        if not byte_values:
            continue

        # Create the linked list for this word
        head = Node(byte_values[0])
        curr_node = head
        for byte in byte_values[1:]:
            new_node = Node(byte, curr_node, None)
            curr_node.next = new_node
            curr_node = curr_node.next
        word_freqs[head] = freq
    
    # --- Step 3: Calculate initial pair statistics ONCE ---
    counts = get_pair_stats(word_freqs)

    # --- Step 4: Optimized Merging Loop ---
    for i in tqdm(range(num_merges), desc="BPE Merges"):
        # Find the pair with the maximum counts and break tie using by choosing the one with greater lexiographical order
        # pair = max(counts, key=counts.get)
        # pair = max(counts, key=lambda p: (counts[p], p))
        pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]
        # --------------------------- 
        del counts[pair]

        # Update index, vocab, and merges
        new_index = len(vocab)
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]]))

        # --- Step 5: Update counts and merge in-place using the linked lists ---
        for word_head, freq in word_freqs.items():
            node = word_head
            while node and node.next:
                # If (A, B) is merged
                if (node.value, node.next.value) == pair:
                    # The count of (X, A) must be decremented.
                    if node.prev:
                        counts[(node.prev.value, node.value)] -= freq
                    # The count of (B, Y) must be decremented.
                    if node.next.next:
                        counts[(node.next.value, node.next.next.value)] -= freq
                    # Set node to be new_index and re-route
                    node.value = new_index
                    if node.next.next:
                        node.next.next.prev = node
                    node.next = node.next.next

                    # The count of (X, M) must be incremented.
                    if node.prev:
                        counts[(node.prev.value, node.value)] += freq
                    # The count of (M, Y) must be incremented.
                    if node.next:
                        counts[(node.value, node.next.value)] += freq

                node = node.next

    return vocab, merges

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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

def train_bpe_tinystories():
    input_file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # --- Start Training and Profiling ---
    start_time = time.monotonic()

    vocab, _ = train_bpe(input_file_path, vocab_size, special_tokens)

    end_time = time.monotonic()
    print(f"Training took: {end_time - start_time:.2f} seconds.")

    # Find and print only the longest token
    longest_token_bytes = max(vocab.values(), key=len)
    print(f"Longest token has {len(longest_token_bytes)} bytes.")
    print(f"Longest token: {longest_token_bytes.decode('utf-8', errors='ignore')}")

def train_bpe_expts_owt():
    input_file_path = "../data/owt_train.txt"
    output_file = "../bpe_owt_params"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    # --- Start Training and Profiling ---
    start_time = time.monotonic()

    vocab, merges = train_bpe(input_file_path, vocab_size, special_tokens)

    end_time = time.monotonic()
    print(f"Training took: {end_time - start_time:.2f} seconds.")

    # Serialize the vocabulary and merges to disk
    params = BPETokenizerParams(vocab=vocab, merges=merges)
    with open(output_file, "wb") as f:
        pickle.dump(params, f)
    print(f"Tokenizer parameters saved to {output_file}")

    # Find and print only the longest token
    longest_token_bytes = max(vocab.values(), key=len)
    print(f"Longest token has {len(longest_token_bytes)} bytes.")
    print(f"Longest token: {longest_token_bytes.decode('utf-8', errors='ignore')}")

def main():
    train_bpe_tinystories()

if __name__ == "__main__":
    main()