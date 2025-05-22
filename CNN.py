import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_class = 1000, dropout = 0.5):
        super(VGG19,self).__init__()
 
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 512*7*7, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(in_features = 4096, out_features = num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

vgg = VGG19()
x = torch.randn(1, 3, 224, 224)
output = vgg(x)
print(output.shape)
