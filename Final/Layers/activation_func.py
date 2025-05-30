from typing import List
import math

class Activation:
    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def softmax(x: List[float]) -> List[float]:
        max_x = max(x)
        exp_x = [math.exp(val - max_x) for val in x]
        sum_exp = sum(exp_x)
        return [exp / sum_exp for exp in exp_x]