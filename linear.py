import torch as t
import torch.nn as nn
import jaxtyping
from utils import tensor_info

t.manual_seed(0)

device = "cuda" if t.cuda.is_available() else "cpu"

# B, C_in, C_out -> Batch In_channels Out_channels
B = 8
C_in = 64
C_out = 128

x = t.randn(B, C_in, requires_grad=True, device=device)
w = nn.Parameter(t.randn(C_out, C_in, requires_grad=True, device=device))
bias = nn.Parameter(t.randn(C_out, requires_grad=True, device=device))

linear = nn.Linear(C_in, C_out)
linear.weight = w
linear.bias = bias

out = linear(x)

dout = t.randn(B, C_out, device=device)
fake_loss = (out * dout).sum()
fake_loss.backward()

tensor_info(x)
tensor_info(w)
tensor_info(out)

# Printing the Gradients of the loss with respect to the inputs
print("x.grad sum :", x.grad.sum().item())
print("w.grad sum :", w.grad.sum().item())
print("bias.grad sum :", bias.grad.sum().item())
