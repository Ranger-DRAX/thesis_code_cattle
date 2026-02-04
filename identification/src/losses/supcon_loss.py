import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos
        
        return loss.mean()


if __name__ == "__main__":
    batch_size, embed_dim = 64, 512
    
    embeddings = torch.randn(batch_size, embed_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([i // 4 for i in range(batch_size)])
    
    loss_fn = SupConLoss(temperature=0.07)
    loss = loss_fn(embeddings, labels)
    
    print(f"SupCon Loss: {loss.item():.4f}")
    
    embeddings.requires_grad = True
    loss = loss_fn(embeddings, labels)
    loss.backward()
    print("Backward pass successful")
