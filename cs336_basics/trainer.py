from jaxtyping import Float, Int
import torch

def cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:
    batch_size = targets.shape[-1]
    max_logit = torch.max(inputs, dim=-1, keepdim=True).values
    return torch.mean(max_logit.squeeze(dim=-1) + torch.log(torch.sum(torch.exp(inputs-max_logit), dim=-1)) - inputs[torch.arange(batch_size), targets])
