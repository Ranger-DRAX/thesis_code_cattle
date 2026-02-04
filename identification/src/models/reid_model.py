import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Literal


class EmbeddingHead(nn.Module):
    def __init__(self, in_features, embed_dim=512, dropout=0.2):
        super(EmbeddingHead, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.bn1 = nn.BatchNorm1d(in_features // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features // 2, embed_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


class ReIDModel(nn.Module):
    def __init__(self, backbone, backbone_out_features, embed_dim=512, dropout=0.2):
        super(ReIDModel, self).__init__()
        self.backbone = backbone
        self.embedding_head = EmbeddingHead(backbone_out_features, embed_dim, dropout)
        self.backbone_frozen = False
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return embeddings
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


class ResNet50ReID(ReIDModel):
    def __init__(self, embed_dim=512, dropout=0.2, pretrained=True):
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        backbone.add_module('flatten', nn.Flatten())
        
        super(ResNet50ReID, self).__init__(backbone, 2048, embed_dim, dropout)
        self.backbone_name = "ResNet-50"


class ConvNeXtTinyReID(ReIDModel):
    def __init__(self, embed_dim=512, dropout=0.2, pretrained=True):
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        convnext = models.convnext_tiny(weights=weights)
        
        backbone = nn.Sequential(convnext.features, convnext.avgpool, nn.Flatten())
        
        super(ConvNeXtTinyReID, self).__init__(backbone, 768, embed_dim, dropout)
        self.backbone_name = "ConvNeXt-Tiny"


def build_reid_model(backbone: Literal["resnet50", "convnext_tiny"],
                     embed_dim=512, dropout=0.2, pretrained=True):
    if backbone == "resnet50":
        return ResNet50ReID(embed_dim, dropout, pretrained)
    elif backbone == "convnext_tiny":
        return ConvNeXtTinyReID(embed_dim, dropout, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


if __name__ == "__main__":
    print("Testing ReID Models")
    
    for backbone in ["resnet50", "convnext_tiny"]:
        model = build_reid_model(backbone, embed_dim=512, dropout=0.2)
        x = torch.randn(4, 3, 224, 224)
        embeddings = model(x)
        
        print(f"\n{backbone.upper()}:")
        print(f"  Total params: {model.get_total_params():,}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  L2 norms: {torch.norm(embeddings, p=2, dim=1)}")
