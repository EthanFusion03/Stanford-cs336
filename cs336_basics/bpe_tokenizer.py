import regex as re
import os
from cs336_basics.bpe_trainer import train_bpe
from typing import Iterator, Iterable
class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
        ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # Create a dictionary for fast merge lookups
        self.merge_map = {pair: i for i, pair in enumerate(self.merges)}
        # Create inverse mapping and special token maps once for efficiency.
        self.byte_to_id = {b:i for i,b in self.vocab.items()}
        self.special_token_ids = {}
        if special_tokens:
            for special_token in special_tokens:
                special_byte = special_token.encode('utf-8', errors='ignore')
                if special_byte in self.byte_to_id:
                    self.special_token_ids[special_token] = self.byte_to_id[special_byte]
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        ''' Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            (in the same format that your BPE training code output) and (optionally) a list of special
            tokens.
        '''
        return NotImplemented
    
    def encode(self, text: str) -> list[int]:
        '''Encode an input text into a sequence of token IDs.'''
        final_ids = []

        # Create a regex pattern to split the text by special tokens, while keeping them.
        if self.special_token_ids:
            sorted_special_ids = sorted(self.special_token_ids.keys(), key=len, reverse=True)
            special_pattern = f"({f'|'.join(re.escape(s) for s in sorted_special_ids)})"
            # Split text into chunks: special tokens and regular text.
            chunks = re.split(special_pattern, text) 
        else:
            # No special tokens, so no splitting needed.
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue

            # If the chunk is a special token, encode it directly and move on.
            if chunk in self.special_token_ids:
                final_ids.append(self.special_token_ids[chunk])
                continue

            # Otherwise, it's regular text that needs pre-tokenization and BPE.
            for match in gpt2_pretokenization(chunk):
                word = match.group(0)

                # 2. Correctly convert pre-token string to a list of single-byte objects.
                # Example: 'the' -> [b't', b'h', b'e']
                byte_parts = [bytes([b]) for b in word.encode('utf-8')]

                # 3. Iteratively apply merges until no more are possible.
                if len(byte_parts) > 0:
                    # Continuously find and apply the highest-priority merge
                    while len(byte_parts) > 1:
                        # Find all pairs in the current word
                        pairs = get_pairs(byte_parts)

                        # Find the best pair to merge (one that appears earliest in self.merges)
                        best_pair = min(pairs, key=lambda x: self.merge_map.get(x, float('inf')))

                        # If no pairs can be merged, we are done with this word
                        if best_pair not in self.merge_map:
                            break

                        # Merge the best pair
                        i = 0
                        new_word_bytes = []
                        while i < len(byte_parts):
                            if i < len(byte_parts)-1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                                # Combine the pair into a single merged byte sequence
                                merged_byte = byte_parts[i] + byte_parts[i+1]
                                new_word_bytes.append(merged_byte)
                                i += 2
                            else:
                                new_word_bytes.append(byte_parts[i])
                                i += 1
                        byte_parts = new_word_bytes
                    
                # Convert the final byte sequences to token IDs
                for b in byte_parts:
                    final_ids.append(self.byte_to_id[b])
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        all_bytes = []
        # 1. Collect all the raw byte sequences for valid token IDs.
        for id in ids:
            if id in self.vocab:
                all_bytes.append(self.vocab[id])
        
        # 2. Join the list of byte sequences into a single bytes object.
        concatenatedBytes = b''.join(all_bytes)
        # 3. Decode the entire byte sequence at once.
        return concatenatedBytes.decode('utf-8', errors='replace')


# Helper Functions below---------------------------------------------------------
def get_pairs(word_bytes: list[bytes]) -> set[tuple[bytes, bytes]]:
    """Helper function to find all adjacent pairs in a byte sequence."""
    return set(zip(word_bytes, word_bytes[1:]))

def gpt2_pretokenization(text: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, text)

def main():
    text = 'the cat ate'
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    example_tokenizer = Tokenizer(vocab, merges)
    ids_list = example_tokenizer.encode(text)
    print(ids_list)

if __name__ == "__main__":
    main()