import numpy as np
from torch.utils.data import Sampler
from typing import Iterator
from collections import defaultdict


class PKSampler(Sampler):
    def __init__(self, data_source, p=16, k=4):
        super(PKSampler, self).__init__(data_source)
        self.data_source = data_source
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        self.id_to_indices = defaultdict(list)
        for idx, label in enumerate(data_source.labels):
            self.id_to_indices[label].append(idx)
        
        self.identities = list(self.id_to_indices.keys())
        self.valid_identities = [
            id_ for id_ in self.identities 
            if len(self.id_to_indices[id_]) >= self.k
        ]
        
        if len(self.valid_identities) < self.p:
            raise ValueError(
                f"Not enough identities with >= {self.k} images. "
                f"Found {len(self.valid_identities)}, need {self.p}"
            )
        
        self.num_batches = len(self.valid_identities) // self.p
        
    def __iter__(self) -> Iterator[int]:
        identities_shuffled = np.random.permutation(self.valid_identities).tolist()
        
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            start_id = batch_idx * self.p
            end_id = start_id + self.p
            batch_identities = identities_shuffled[start_id:end_id]
            
            for identity in batch_identities:
                id_indices = self.id_to_indices[identity]
                
                if len(id_indices) == self.k:
                    sampled_indices = id_indices
                else:
                    sampled_indices = np.random.choice(
                        id_indices, size=self.k, replace=False
                    ).tolist()
                
                batch_indices.extend(sampled_indices)
            
            for idx in batch_indices:
                yield idx
    
    def __len__(self):
        return self.num_batches * self.batch_size


class PKBatchSampler:
    def __init__(self, dataset, p=16, k=4):
        self.sampler = PKSampler(dataset, p, k)
        self.batch_size = p * k
        
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
    
    def __len__(self):
        return len(self.sampler) // self.batch_size


if __name__ == "__main__":
    class DummyDataset:
        def __init__(self):
            self.labels = [i // 4 for i in range(200)]
    
    dataset = DummyDataset()
    sampler = PKBatchSampler(dataset, p=16, k=4)
    
    print(f"Dataset: {len(dataset.labels)} samples")
    print(f"Unique IDs: {len(set(dataset.labels))}")
    print(f"Batches per epoch: {len(sampler)}")
    print(f"Batch size: {sampler.batch_size}")
    
    for i, batch in enumerate(sampler):
        if i == 0:
            labels = [dataset.labels[idx] for idx in batch]
            print(f"\nFirst batch labels: {labels[:20]}...")
            print(f"Unique IDs in batch: {len(set(labels))}")
        break
