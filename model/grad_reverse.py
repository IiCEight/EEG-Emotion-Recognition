from typing import Any, Tuple

import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

class GradientReverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha):
        return GradientReverseFunction.apply(x, alpha)
    

class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: float = 1.0, low: float = 0.0, high: float = 1.,
                 max_iters: int = 1000., auto_step: bool = False):
        super().__init__()
        self.alpha = alpha
        self.low = low
        self.high = high
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float64(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.high - self.low) + self.low
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

    

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, reverse_coeff: float = 1.0) -> torch.Tensor:
        ctx.reverse_coeff = reverse_coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.reverse_coeff, None
