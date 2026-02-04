import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, margin=0.30, scale=30.0):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, labels):
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return self.ce_loss(output, labels)


if __name__ == "__main__":
    batch_size, embed_dim, num_classes = 64, 512, 190
    
    embeddings = torch.randn(batch_size, embed_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([i % 16 for i in range(batch_size)])
    
    loss_fn = ArcFaceLoss(num_classes, embed_dim, margin=0.30, scale=30.0)
    loss = loss_fn(embeddings, labels)
    
    print(f"ArcFace Loss: {loss.item():.4f}")
    
    embeddings.requires_grad = True
    loss = loss_fn(embeddings, labels)
    loss.backward()
    print("Backward pass successful")
