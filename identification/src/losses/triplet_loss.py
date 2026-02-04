import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.30):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        labels_equal = labels_equal & ~diagonal_mask
        
        losses = []
        for i in range(batch_size):
            positive_mask = labels_equal[i]
            if not positive_mask.any():
                continue
            
            hardest_positive_dist = pairwise_dist[i][positive_mask].max()
            
            negative_mask = labels_not_equal[i]
            if not negative_mask.any():
                continue
            
            hardest_negative_dist = pairwise_dist[i][negative_mask].min()
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return torch.stack(losses).mean()


if __name__ == "__main__":
    batch_size, embed_dim = 64, 512
    
    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([i // 4 for i in range(batch_size)])
    
    loss_fn = TripletLoss(margin=0.30)
    loss = loss_fn(embeddings, labels)
    
    print(f"Triplet Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass successful")
