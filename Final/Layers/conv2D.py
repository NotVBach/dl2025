from .random_lite import Random as rand
from typing import List, Tuple
from .layer import Layer
from .activation_func import Activation as act


class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: Tuple[int, int], stride: int, padding: str, activation: str):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.lower() == 'same'
        self.activation = activation.lower()
        self.weights = None
        self.bias = None
        self.input_shape = None
        self.output_shape = None
    
    def initialize(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        in_channels, in_height, in_width = input_shape
        k_h, k_w = self.kernel_size
        rng = rand()
        '''
        The 4-D weights:
            Height (int): k_h
            Width (int): k_w
            Input Channels (int): in_channels
            Filters (int): filters
        '''
        self.weights = [[[[rng.gauss(0, 0.1) 
                           for _ in range(k_h)] 
                          for _ in range(k_w)]
                         for _ in range(in_channels)] 
                         for _ in range(self.filters)]
        
        self.bias = [0.0 for _ in range(self.filters)]
        out_height = in_height if self.padding else (in_height - k_h + 1) // self.stride
        out_width = in_width if self.padding else (in_width - k_w + 1) // self.stride
        self.output_shape = (self.filters, out_height, out_width)
    
    def forward(self, input_data: List[List[List[float]]]) -> List[List[List[float]]]:
        in_channels, in_height, in_width = self.input_shape
        k_h, k_w = self.kernel_size
        _, out_height, out_width = self.output_shape
        output = [[[0.0 for _ in range(out_width)] for _ in range(out_height)] for _ in range(self.filters)]
        
        for f in range(self.filters):
            for i in range(0, out_height * self.stride, self.stride):
                for j in range(0, out_width * self.stride, self.stride):
                    sum_val = self.bias[f]
                    for c in range(in_channels):
                        for ki in range(k_h):
                            for kj in range(k_w):
                                in_i = i + ki - (k_h // 2 if self.padding else 0)
                                in_j = j + kj - (k_w // 2 if self.padding else 0)
                                if 0 <= in_i < in_height and 0 <= in_j < in_width:
                                    sum_val += input_data[c][in_i][in_j] * self.weights[f][c][ki][kj]
                    output[f][i // self.stride][j // self.stride] = (
                        act.relu(sum_val) if self.activation == 'relu' else sum_val
                    )
        return output
    
    def backward(self, grad_output: List[List[List[float]]], learning_rate: float) -> List[List[List[float]]]:
        in_channels, in_height, in_width = self.input_shape
        grad_input = [[[0.0 for _ in range(in_width)] for _ in range(in_height)] for _ in range(in_channels)]
        return grad_input  # Simplified, full backprop requires gradient computation
