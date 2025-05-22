class Node:
    def __init__(self):
        self.bias = 0.0   
        self.weights = [] 
    
class Layer:
    def __init__(self, node_num):
        self.nodes = [Node() for _ in range(node_num)]

class NN:
    def __init__(self, layer_num):
        self.layers = [Layer() for _ in range(layer_num)]

def readConfig(path):
    config = []
    with open(path, 'r') as file:
        # next(file)
        for line in file:
            value = line.strip()
            value = float(value) 
            config.append(value)
    layerNum = int(config[0])
    config.pop(0)
    return layerNum, config

def uniRand():
    seed = None
    if seed is None:
        temp = object()
        pseudo_seed = id(temp) ^ 0xDEADBEEF  
        seed = pseudo_seed % (2**31) 
    
    a = 1103515245
    c = 12345
    m = 2**31

    seed = (a * seed + c) % m
    # Normalize between 0 and 1
    return (seed % 10000) / 10000

layerNum, config = readConfig("resource/Lab3/config.txt")
print(layerNum, config)




    