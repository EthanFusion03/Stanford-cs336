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
        

