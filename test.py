import torch
import torch.nn.functional as F

Q = torch.randn(1, 1, 4, 64)  # 1 query
K_valid = torch.randn(1, 3, 4, 64)  # 3 valid keys
V_valid = torch.randn(1, 3, 4, 64)

# Without zeros
scores1 = (Q @ K_valid.transpose(-2, -1)) / (64**0.5)
attn1 = F.softmax(scores1, dim=-1)
out1 = attn1 @ V_valid

# With zeros (10 zero keys)
K_with_zeros = torch.cat([torch.zeros(1, 9, 4, 64), K_valid], dim=1)
V_with_zeros = torch.cat([torch.zeros(1, 9, 4, 64), V_valid], dim=1)
scores2 = (Q @ K_with_zeros.transpose(-2, -1)) / (64**0.5)
attn2 = F.softmax(scores2, dim=-1)
out2 = attn2 @ V_with_zeros

print("Attention weights (no zeros):", attn1)
print("Attention weights (with zeros):", attn2[0, -3:])
print("attn1 shape:", attn1.shape)
print("attn2 shape:", attn2.shape)
print("out1 shape:", out1.shape)
print("out2 shape:", out2.shape)
# out1 has 3 elements in dim 1, out2 has 12 (9 zeros + 3 valid)
# Compare out1 with the last 3 elements of out2 (the valid keys part)
# print("Output difference:", (out1 - out2[:, -3:]).abs().max().item())
print(out2)
