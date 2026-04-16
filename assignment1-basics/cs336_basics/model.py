from multiprocessing.connection import answer_challenge
from sympy import true
import torch 
import torch.nn as nn
import math 
from einops import rearrange,repeat

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self,num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_value = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        rmsnorm = x / rms_value * self.weight
        return rmsnorm.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x))*self.w3(x))

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float , d_k : int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len, device=device,dtype = torch.float32)
        freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype = torch.float32) / d_k)
        angles = torch.outer(positions, freq)
        self.register_buffer("cos", angles.cos(),persistent=False)
        self.register_buffer("sin", angles.sin(),persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_odd = x[..., 1::2]
        x_even = x[..., 0::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_odd * cos + x_even * sin
        return output

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_num = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - max_num
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    output = torch.einsum("...qk,...kd->...qd", attn, V)
    return output

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, max_seq_len: int = 2048, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionEmbedding(theta, self.head_dim, self.max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor, use_rope: bool) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x) # (batch_size, sequence_length, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        if use_rope:
            if token_positions is None:
                token_positions = repeat(torch.arange(seq_len, device=x.device), "s -> b s", b = batch_size)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        y = scaled_dot_product_attention(q, k, v, causal_mask)
        y = rearrange(y, "b h s d -> b s (h d)")
        return self.output_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = 10000.0)->None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x),token_positions=None, use_rope=true)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0)->None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_shift = logits - logits.amax(dim=-1, keepdim=True)
    log_probs = torch.log(torch.exp(logits_shift).sum(dim=-1, keepdim=True))
    target_logits = logits_shift.gather(dim=-1, index=targets.unsqueeze(-1))
    return (log_probs - target_logits).mean()
