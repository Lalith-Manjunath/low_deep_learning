import torch as t
import torch.nn as nn
import jaxtyping

t.manual_seed(0)

device = "cuda" if t.cuda.is_avaliable() else "cpu"

# B, C_in, C_out -> Batch In_channels Out_channels
B = 8
C_in = 64
C_out = 128

x = t.randn(B, C_in, requires_grad=True, device=device)
print(x)
