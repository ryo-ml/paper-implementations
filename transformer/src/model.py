import torch
import torch.nn as nn

from .mask import *

class CrossAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 8,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"{d_model} must be divisible by {num_heads}"

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor, # (B, L_q, d_model)
        k: torch.Tensor | None = None, # (B, L_kv, d_model)
        v: torch.Tensor | None = None, # (B, L_kv, d_model)
        mask: torch.BoolTensor | None = None, # (B, 1, L_q, L_kv)
    ) -> torch.Tensor:
        if k is None:
            k = q
        if v is None:
            v = q

        B, L_q, _ = q.size()
        _, L_kv, _ = k.size()

        q = self.proj_q(q).view(B, L_q, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (B, H, L_q, d_k)
        k = self.proj_k(k).view(B, L_kv, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (B, H, L_kv, d_k)
        v = self.proj_v(v).view(B, L_kv, self.num_heads, self.d_v).permute(0, 2, 1, 3) # (B, H, L_kv, d_v)

        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) * (self.d_k ** -0.5) # (B, H, L_q, L_kv)

        if mask is not None:
            scores = scores.masked_fill(mask, -torch.inf)

        attn_weights = self.softmax(scores) # (B, H, L_q, L_kv)
        attn_output = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model) # (B, L_q, d_model)

        out = self.proj_o(attn_output) # (B, L_q, d_model)
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        p: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super().__init__()

        # multi-head attention
        self.self_attn = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.dropout= nn.Dropout(p)
        self.ln1 = nn.LayerNorm(d_model)

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            src_mask: torch.BoolTensor,
        ) -> torch.Tensor:
        x = self.ln1(x + self.dropout(self.self_attn(x, mask=src_mask)))
        x = self.ln2(x + self.dropout(self.feed_forward(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        p: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super().__init__()

        # multi-head attention
        self.self_attn = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.dropout= nn.Dropout(p)
        self.ln1 = nn.LayerNorm(d_model)

        # multi-head attention
        self.cross_attn = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.ln2 = nn.LayerNorm(d_model)

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ln1(x + self.dropout(self.self_attn(x, mask=tgt_mask)))
        x = self.ln2(x + self.dropout(self.cross_attn(x, k, v, mask=src_mask)))
        x = self.ln3(x + self.dropout(self.feed_forward(x)))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        max_len: int = 512,
        d_model: int = 512,
    ) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        p: float = 0.1,
        pe: nn.Module = nn.Module,
        max_len: int = 512,
        n_encoder: int = 6,
        n_decoder: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = pe
        self.dropout = nn.Dropout(p)

        # encoder
        encoder = []
        for _ in range(n_encoder):
            encoder.append(
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    p=p,
                    d_ff=d_ff,
                )
            )

        # decoder
        decoder = []
        for _ in range(n_decoder):
            decoder.append(
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    p=p,
                    d_ff=d_ff,
                )
            )

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = self.embedding(src) * (self.d_model ** 0.5)
        src = self.dropout(self.pe(src))

        for layer in self.encoder:
            src = layer(
                x=src,
                src_mask=src_mask
            ) 
        return src

    def decode(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        tgt = self.dropout(self.pe(tgt))

        for layer in self.decoder:
            tgt = layer(
                x=tgt,
                k=src,
                v=src,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
            )

        out = self.fc(tgt)
        return out

    def forward(
        self,
        src: torch.Tensor, # (B, L_src)
        tgt: torch.Tensor, # (B, L_tgt)
    ) -> torch.Tensor:
        src_mask = make_padding_mask(src, self.pad_token_id)
        tgt_pad_mask = make_padding_mask(tgt, self.pad_token_id)
        tgt_causal_mask = make_causal_mask(tgt.size(1), tgt.device)
        tgt_mask = tgt_causal_mask | tgt_pad_mask

        src = self.encode(
            src=src,
            src_mask=src_mask,
        )

        out = self.decode(
            tgt=tgt,
            src=src,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
        )

        return out

    def generate(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        src_mask = make_padding_mask(src, self.pad_token_id)

        with torch.no_grad():
            src = self.encode(
                src=src,
                src_mask=src_mask,
            )

        seq = torch.full((src.size(0), 1), self.bos_token_id, dtype=torch.long, device=src.device) # (B, 1)
        print(seq)

        for _ in range(1, self.max_len):
            tgt_pad_mask = make_padding_mask(seq, self.pad_token_id)
            tgt_causal_mask = make_causal_mask(seq.size(1), seq.device)
            tgt_mask = tgt_pad_mask | tgt_causal_mask

            logits = self.decode(
                tgt=seq,
                src=src,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
            ) # (B, L_q, vocab_size)

            prob = logits[:, -1, :] # (B, vocab_size)
            next_token = torch.argmax(prob, dim=-1, keepdim=True) # (B, 1)
            seq = torch.cat([seq, next_token], dim=1)

            if (next_token == self.eos_token_id).all():
                break

        return seq



if __name__ == '__main__':
    pe = SinusoidalPositionalEncoding(max_len=512, d_model=512)

    transformer = Transformer(
        vocab_size=1000,
        d_model=512,
        p=0.1,
        pe=pe,
        max_len=512,
        n_encoder=6,
        n_decoder=6,
        num_heads=8,
        d_ff=2048,
    )

    src = torch.randint(0, 1000, (32, 20))
    tgt = torch.randint(0, 1000, (32, 10))
    logits = transformer(src, tgt)

    print(f"Input Shape: {src.shape}")   # [32, 20]
    print(f"Target Shape: {tgt.shape}")  # [32, 10]
    print(f"Logits Shape: {logits.shape}") # [32, 10, 1000]
