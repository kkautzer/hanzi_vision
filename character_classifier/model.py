import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChineseCharacterCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Convolutional Neural Network for Chinese Character Classification, based on GoogLeNet.

        Parameters:
            num_classes (int): Number of unique Chinese characters (classes) to classify.
        """
        super(ChineseCharacterCNN, self).__init__()
        
        ## base this model off of GoogLeNet
        self.googlenet = models.googlenet()
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, out_features=num_classes, bias=True)
        # modify # of output classes (from 1000 to num_classes)
        print(self.googlenet.fc)
        
    def forward(self, x):
        """
        Forward pass of the CNN.
        
        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 1, 64, 64)
        Returns:
            Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.googlenet(x)

# Example usage for testing
if __name__ == "__main__":
    model = ChineseCharacterCNN(num_classes=3928)    
    # print(model)
    # dummy_input = torch.randn(8, 1, 64, 64)  # Example input (batch of 8 grayscale 64x64 images)
    # output = model(dummy_input)
    # print('---')
    # print(f"Output shape: {output.shape}")  # Should print torch.Size([8, 100])
    # print(f"Output: {output}")
