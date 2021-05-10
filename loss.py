import torch
import torch.nn as nn
import torch.nn.functional as F

def negative_cosine_similarity(
    p: torch.Tensor,
    z: torch.Tensor
) -> torch.Tensor:
    """ D(p, z) = -(p*z).sum(dim=1).mean() """
    z = z.detach() # stop gradient
    p = F.normalize(p, dim=1) # l2-normalize 
    z = F.normalize(z, dim=1) # l2-normalize 
    return -(p*z).sum(dim=1).mean()