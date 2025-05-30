from .random_lite import Random as rand
from typing import List, Tuple
from .layer import Layer
from .activation_func import Activation as act

class Dense(Layer):
    def __init__(self, units: int, activation: str = 'none'):
        self.units = units
        self.activation = activation.lower()
        self.weights = None
        self.bias = None
        self.input_shape = None
        self.output_shape = (units,)
        self.input_data = None
        self.output = None
    
    def initialize(self, input_shape: Tuple[int]):
        self.input_shape = input_shape
        in_features = input_shape[0]
        rng = rand()
        self.weights = [[rng.gauss(0, 0.1) for _ in range(in_features)] for _ in range(self.units)]
        self.bias = [0.0 for _ in range(self.units)]
    
    def forward(self, input_data: List[float]) -> List[float]:
        self.input_data = input_data
        self.output = []
        for i in range(self.units):
            sum_val = self.bias[i]
            for j in range(len(input_data)):
                sum_val += input_data[j] * self.weights[i][j]
            self.output.append(sum_val)
        if self.activation == 'relu':
            self.output = [act.relu(x) for x in self.output]
        elif self.activation == 'softmax':
            self.output = act.softmax(self.output)
        return self.output
    
    def backward(self, grad_output: List[float], learning_rate: float) -> List[float]:
        grad_input = [0.0 for _ in range(self.input_shape[0])]
        for i in range(self.units):
            grad = grad_output[i] * (act.relu_derivative(self.output[i]) if self.activation == 'relu' else 1.0)
            for j in range(self.input_shape[0]):
                self.weights[i][j] -= learning_rate * grad * self.input_data[j]
                grad_input[j] += grad * self.weights[i][j]
            self.bias[i] -= learning_rate * grad
        return grad_input
