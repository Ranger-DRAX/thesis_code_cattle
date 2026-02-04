import torch
import torch.nn as nn
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from losses.supcon_loss import SupConLoss
from losses.arcface_loss import ArcFaceLoss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, supcon_temperature=0.07,
                 arcface_margin=0.30, arcface_scale=30.0, 
                 weight_supcon=1.0, weight_arcface=1.0):
        super(CombinedLoss, self).__init__()
        
        self.supcon_loss = SupConLoss(temperature=supcon_temperature)
        self.arcface_loss = ArcFaceLoss(num_classes, embedding_dim, 
                                       arcface_margin, arcface_scale)
        
        self.weight_supcon = weight_supcon
        self.weight_arcface = weight_arcface
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
    
    def forward(self, embeddings, labels):
        supcon_val = self.supcon_loss(embeddings, labels)
        arcface_val = self.arcface_loss(embeddings, labels)
        
        total = self.weight_supcon * supcon_val + self.weight_arcface * arcface_val
        
        loss_dict = {
            'total': total.item(),
            'supcon': supcon_val.item(),
            'arcface': arcface_val.item()
        }
        
        return total, loss_dict


if __name__ == "__main__":
    import torch.nn.functional as F
    
    batch_size, embed_dim, num_classes = 64, 512, 190
    
    embeddings = torch.randn(batch_size, embed_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([i % 16 for i in range(batch_size)])
    
    loss_fn = CombinedLoss(num_classes, embed_dim, weight_supcon=1.0, weight_arcface=1.0)
    total_loss, loss_dict = loss_fn(embeddings, labels)
    
    print(f"SupCon: {loss_dict['supcon']:.4f}")
    print(f"ArcFace: {loss_dict['arcface']:.4f}")
    print(f"Total: {loss_dict['total']:.4f}")
    
    embeddings.requires_grad = True
    total_loss, _ = loss_fn(embeddings, labels)
    total_loss.backward()
    print("Backward pass successful")
