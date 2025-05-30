from typing import List, Tuple
from .layer import Layer


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
    
    def initialize(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] * input_shape[1] * input_shape[2],)
    
    def forward(self, input_data: List[List[List[float]]]) -> List[float]:
        in_channels, in_height, in_width = self.input_shape
        output = []
        for c in range(in_channels):
            for i in range(in_height):
                for j in range(in_width):
                    output.append(input_data[c][i][j])
        return output
    
    def backward(self, grad_output: List[float], learning_rate: float) -> List[List[List[float]]]:
        in_channels, in_height, in_width = self.input_shape
        grad_input = [[[0.0 for _ in range(in_width)] for _ in range(in_height)] for _ in range(in_channels)]
        idx = 0
        for c in range(in_channels):
            for i in range(in_height):
                for j in range(in_width):
                    grad_input[c][i][j] = grad_output[idx]
                    idx += 1
        return grad_input