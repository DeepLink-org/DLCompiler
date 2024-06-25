
import torch

'''
a lightweight IR for dlBLAS
mostly we rely on pytorch2' implmentation
to convert a torch.Tensor -> FakeTensor -> inductor IR
which then can be put into templates
'''