from typing import List, Tuple
from .layer import Layer

class MaxPool2D(Layer):
    def __init__(self, pool_size: Tuple[int, int], stride: int):
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = None
        self.output_shape = None
    
    def initialize(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        in_channels, in_height, in_width = input_shape
        p_h, p_w = self.pool_size
        out_height = (in_height - p_h) // self.stride + 1
        out_width = (in_width - p_w) // self.stride + 1
        self.output_shape = (in_channels, out_height, out_width)
    
    def forward(self, input_data: List[List[List[float]]]) -> List[List[List[float]]]:
        in_channels, in_height, in_width = self.input_shape
        _, out_height, out_width = self.output_shape
        p_h, p_w = self.pool_size
        output = [[[0.0 for _ in range(out_width)] for _ in range(out_height)] for _ in range(in_channels)]
        
        for c in range(in_channels):
            for i in range(0, in_height - p_h + 1, self.stride):
                for j in range(0, in_width - p_w + 1, self.stride):
                    max_val = float('-inf')
                    for pi in range(p_h):
                        for pj in range(p_w):
                            max_val = max(max_val, input_data[c][i + pi][j + pj])
                    output[c][i // self.stride][j // self.stride] = max_val
        return output
    
    def backward(self, grad_output: List[List[List[float]]], learning_rate: float) -> List[List[List[float]]]:
        in_channels, in_height, in_width = self.input_shape
        grad_input = [[[0.0 for _ in range(in_width)] for _ in range(in_height)] for _ in range(in_channels)]
        return grad_input