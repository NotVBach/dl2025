from typing import List, Tuple
import math
from PIL import Image
import matplotlib.pyplot as plt
from Layers import maxpool2D, conv2D, flatten, dense


class CNN:
    def __init__(self, config_file: str):
        self.layers = []
        self.input_shape = None
        self.loss_history = []
        self.parse_config(config_file)
    
    def parse_config(self, config_file: str):
        with open(config_file, 'r') as f:
            lines = f.readlines()
        num_layers = int(lines[0].strip())
        
        # Get layer count for debug
        conv_count = 0
        maxpool_count = 0
        dense_count = 0
        
        for i, line in enumerate(lines[1:num_layers+1]):
            parts = line.strip().split()
            layer_type = parts[0]
            params = dict([p.split('=') for p in parts[1:]] if len(parts) > 1 else [])
            
            if layer_type == 'Input':
                if i != 0:
                    raise ValueError("Input layer must be first") # Ensure input layer is in the beginning
                dims = list(map(int, params['shape'].split('x')))
                if len(dims) != 3:
                    raise ValueError("Input shape must be CxHxW")
                self.input_shape = (dims[0], dims[1], dims[2])  # (channels, height, width)
                continue
            
            if layer_type == 'Conv2D':
                conv_count += 1
                filters = int(params['filters'])
                kernel = tuple(map(int, params['kernel'].split('x')))
                stride = int(params.get('stride', 1))
                padding = params.get('padding', 'valid')
                activation = params.get('activation', 'relu')
                self.layers.append(conv2D.Conv2D(filters, kernel, stride, padding, activation))

            elif layer_type == 'MaxPool2D':
                maxpool_count += 1
                pool = tuple(map(int, params['pool'].split('x')))
                stride = int(params.get('stride', pool[0]))
                self.layers.append(maxpool2D.MaxPool2D(pool, stride))

            elif layer_type == 'Dense':
                dense_count += 1
                units = int(params['units'])
                activation = params.get('activation', 'softmax' if i == num_layers-1 else 'relu')
                # Insert flatten before the first dense layer
                if dense_count == 1:
                    self.layers.append(flatten.Flatten())
                self.layers.append(dense.Dense(units, activation))
    
    def initialize(self, input_shape: Tuple[int, int, int]):
        current_shape = input_shape
        for layer in self.layers:
            layer.initialize(current_shape)
            current_shape = layer.output_shape
    
    def forward(self, input_data: List[List[List[float]]]) -> List[float]:
        data = input_data
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def train(self, images: List[List[List[float]]], labels: List[int], epochs: int, learning_rate: float):
        self.initialize(input_shape=(1, 28, 28))
        self.loss_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for img, label in zip(images, labels):
                output = self.forward(img)
                target = [1.0 if i == label else 0.0 for i in range(2)]
                loss = -sum(t * math.log(max(o, 1e-10)) for t, o in zip(target, output))
                total_loss += loss
                grad_output = [o - t for o, t in zip(output, target)]

                for layer in reversed(self.layers):
                    grad_output = layer.backward(grad_output, learning_rate)

            avg_loss = total_loss / len(images)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def load_image(self, image_path: str) -> List[List[List[float]]]:
        # Grayscale to reduce amount of work
        img = Image.open(image_path).convert('L')
        img_data = [[float(img.getpixel((i, j))) / 255.0 
                     for j in range(img.size[1])]
                    for i in range(img.size[0])]
        return [img_data]
    
    def load_folder(self, image_folder: str, label_file: str) -> Tuple[List[List[List[float]]], List[int]]:
        images = []
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                image_path = f"{image_folder}/{filename}"
                img_data = self.load_image(image_path)
                images.append(img_data)
                labels.append(int(label))
        return images, labels

        
    def predict_image(self, image_data: List[List[List[float]]]) -> int:
        prediction = self.forward(image_data)
        return prediction.index(max(prediction))
    
    def plot_loss(self):
        if not self.loss_history:
            print("What do you expect me to plot without training ??")
            return
        else:
            plt.figure()
            plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig("plots/Train_over_loss.jpg")
            plt.show()
    
    def predict_image(self, image_path: str) -> int:
        img_data = self.load_single_image(image_path)
        prediction = self.forward(img_data)
        return prediction.index(max(prediction))
    
    def predict_folder(self, image_folder: str, label_file: str) -> Tuple[float, List[List[int]]]:
        images, labels = self.load_folder(image_folder, label_file)
        correct = 0
        total = len(images)
        confusion_matrix = [[0, 0], [0, 0]]
        
        for img, label in zip(images, labels):
            prediction = self.forward(img)
            predicted_class = prediction.index(max(prediction))
            confusion_matrix[label][predicted_class] += 1
            if predicted_class == label:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, confusion_matrix
    
    def plot_confusion_matrix(self, confusion_matrix: List[List[int]], classes: List[str] = ['0', '1']):
        plt.figure(figsize=(6, 6))
        max_value = max(max(row) for row in confusion_matrix)
        im = plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)
        
        plt.title('Confusion Matrix')
        tick_marks = list(range(len(classes)))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        thresh = max_value / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(confusion_matrix[i][j]),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i][j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("plots/Confusion_Matrix.jpg")
        plt.show()
