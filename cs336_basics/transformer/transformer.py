import torch
from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.swiglu import swiglu
from cs336_basics.transformer.rmsnorm import Rmsnorm
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.softmax import softmax

class prenorm_XformerBlock(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            d_ff: int,
            use_rope: bool = True,
            max_seq_len: int = 2048, 
            theta: int = 10000.0,
    ):
        super().__init__()
        self.attn_layer = MultiHeadSelfAttention(d_model, num_heads, use_rope=use_rope, max_seq_len=max_seq_len, theta=theta)
        self.ffn_layer = swiglu(d_model, d_ff)
        self.norm1 = Rmsnorm(d_model)
        self.norm2 = Rmsnorm(d_model)
    
    # x has input dim: (batch_size, seq_len, d_model)
    def forward(self, x):
        # Attention prcoess step
        x = x + self.attn_layer(self.norm1(x))
        # FFN step
        x = x + self.ffn_layer(self.norm2(x))
        return x

class Xformer_LM(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int, # max_seq_len for RoPE
        d_model: int, # embedding size
        num_layers: int, # num of xformer blocks
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.emb_layer = Embedding(vocab_size, d_model)
        self.tfr_blocks = torch.nn.ModuleList(
            [prenorm_XformerBlock(d_model, num_heads, d_ff, True, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.norm_layer = Rmsnorm(d_model)
        self.linear_layer = Linear(d_model, vocab_size)
    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Transformer language model.

        Args:
            token_ids: The input tensor of shape (batch_size, seq_len).

        Returns:
            The output tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Get token embeddings
        x = self.emb_layer(in_indices)

        # Pass through # of prenorm-transformer blocks
        for layer in self.tfr_blocks:
            x = layer(x)

        # Normalize again
        x = self.norm_layer(x)
        # Linear layer
        x = self.linear_layer(x)
        # return softmax(x)
        return x
      

def decode(self,
        prompt: torch.Tensor,
        max_gen_tokens: int,
        tau: float = 1.0,
        top_p: float = 0.9,
        end_of_txt_id: int = -1
    ):
        """
        Generates text auto-regressively using temperature and top-p (nucleus) sampling.
        
        Args:
            prompt: The initial sequence of token_ids, shape (batch_size, seq_len).
            max_gen_tokens: The maximum number of tokens to generate.
            tau: The temperature for softmax scaling. Higher is more random.
            top_p: The nucleus sampling threshold.
            end_of_txt_id: The token_id that signals the end of generation.
        """
        generated = prompt
        # We assume batch_size is 1 for this implementation
        assert generated.shape[0] == 1, "This decode implementation only supports a batch size of 1."

        for _ in range(max_gen_tokens):
            # --- FIX: Ensure the input sequence doesn't exceed the model's context length ---
            # Crop the context if it's too long
            current_context = generated[:, -self.context_length:]

            # Get the logits for the very last token in the sequence
            logits = self.forward(current_context)[:, -1, :] # Shape: (1, vocab_size)

            # Apply temperature scaling
            logits = logits / tau
            
            # Get probabilities
            probs = softmax(logits, dim=-1) # Shape: (1, vocab_size)
            
            # --- Efficient Top-p (Nucleus) Sampling ---
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create a mask to remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create a mask for the original indices
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )

            # Zero out the probabilities of tokens to remove
            probs[indices_to_remove] = 0.0
            
            # Re-normalize the remaining probabilities
            probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1) # Shape: (1, 1)
            # Append the new token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Stop if end-of-text token is generated
            if next_token == end_of_txt_id:
                break

        return generated
