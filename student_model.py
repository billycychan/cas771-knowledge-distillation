import torch
import torch.nn as nn
import torchvision.models as models
import timm

class MobileNetV3Student(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        """
        MobileNetV3 model with pre-trained ImageNet weights for knowledge distillation
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights from ImageNet
        """
        super().__init__()
        
        # Load pre-trained MobileNetV3-Small which works well with small images
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Replace the classifier with our own for the target number of classes
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes]
        """
        return self.backbone(x)


class EfficientNetStudent(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        """
        EfficientNet-B0 model with pre-trained ImageNet weights
        Optimized for small image sizes while maintaining good accuracy
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights from ImageNet
        """
        super().__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace classifier head for our task
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)


class ResNet18Student(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        """
        ResNet18 model with pre-trained ImageNet weights
        A lightweight but powerful model that works well with smaller images
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights from ImageNet
        """
        super().__init__()
        
        # Load pre-trained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)


# Default model class to use for knowledge distillation
# Choose the one you want to use as default here
# StudentModel = MobileNetV3Student
StudentModel = EfficientNetStudent
# StudentModel = ResNet18Student
