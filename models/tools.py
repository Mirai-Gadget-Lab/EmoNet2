import torch

def create_negative_samples(pooled_txt_feat: torch.Tensor, pos_idx: int = 0):
    
    x = pooled_txt_feat.unsqueeze(1) # (bs, 1, hideen_size)
    
    # Create negative sample using shift
    # Number of negative sample = 4
    negative_samples = torch.cat([
        x.roll(shifts=1, dims=0)[:,:2,:],
        x.roll(shifts=2, dims=0)[:,:2,:],
        x.roll(shifts=-1, dims=0)[:,:2,:],
        x.roll(shifts=-2, dims=0)[:,:2,:]
    ], dim=1)
    output = torch.cat([x, negative_samples], dim=1) # (bs, 5, hideen_size)
    
    # Shuffle negative samples and positive samples
    idx = torch.randperm(5)
    output = output[:, idx].view(output.size())
    pos_idx = (idx == pos_idx).nonzero(as_tuple=True)[0].item()
    output[:, pos_idx] = pooled_txt_feat
    
    labels = torch.zeros(output.shape[:2]).type_as(output)
    labels[:, pos_idx] = 1
    assert torch.all(output[:, pos_idx] == pooled_txt_feat)
    
    return output, labels