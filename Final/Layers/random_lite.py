import math

class Random:
    '''
    Randomly select a random function from the internet then take it as reference
    Absolutely not copy and paste
    '''

    def __init__(self, seed: int = 666666):
        self.seed = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32
    
    def _next(self) -> int:
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed
    
    def uniform(self) -> float:
        return self._next() / self.m
    
    def gauss(self, mean: float = 0.0, std: float = 1.0) -> float:
        u1 = self.uniform()
        u2 = self.uniform()

        if u1 == 0:
            u1 = 1e-10

        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z