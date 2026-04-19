import torch

def make_padding_mask(seq: torch.Tensor, pad_token_id: int) -> torch.BoolTensor:
    '''
    seq: (B, L)
    '''
    return (seq == pad_token_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(seq_len: int, device: torch.device) -> torch.BoolTensor:
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
    return causal_mask.unsqueeze(0).unsqueeze(0)

if __name__ == '__main__':
    seq = torch.tensor([[2, 4, 3, 0]])
    pad_mask = make_padding_mask(seq, pad_token_id=0)
    causal_mask = make_causal_mask(seq_len=4)
    print(f'padding mask: {pad_mask.shape}\n{pad_mask}')
    print()
    print(f'causal mask: {causal_mask.shape}\n{causal_mask}')