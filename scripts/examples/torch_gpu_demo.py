import torch

print (torch.__version__)
print (torch.cuda.is_available())

# Manually place operations on GPU:
print(torch.rand(2,3).cuda())
