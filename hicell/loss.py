
import torch
import torch.nn as nn
import torch.nn.functional as F

class GInfLoss(nn.Module):
    def __init__(self, class_counts, mu=0.0, sigma=0.1, eps=1e-6):
        """
        Gaussian Inflation Loss for long-tailed recognition (Enhanced Version)
        
        Args:
            class_counts (Tensor): Number of samples per class [num_classes]
            mu (float): Mean of Gaussian distribution (default: 0.0)
            sigma (float): Std dev of Gaussian noise (default: 0.1)
            eps (float): Numerical stabilizer (default: 1e-6)
        """
        super().__init__()
        # Parameter validation
        assert torch.all(class_counts > 0), "Class counts must be positive"
        assert sigma >= 0, "Sigma must be non-negative"
        
        self.num_classes = class_counts.shape[0]
        self.mu = mu
        self.sigma = sigma
        self.eps = eps
        
        # Register persistent buffers
        self.register_buffer('Nmax', class_counts.max().float())
        self.register_buffer('delta_base', self._compute_delta_base(class_counts))
        
    def _compute_delta_base(self, counts):
        """Safely compute inflation base values with numerical stability"""
        return torch.log(self.Nmax + self.eps) - torch.log(counts.float() + self.eps)

    def forward(self, logits, targets):
        """
        Forward pass with dynamic inflation mechanism
        
        Args:
            logits (Tensor): Model outputs [batch_size, num_classes]
            targets (Tensor): Ground truth labels [batch_size]
            
        Returns:
            loss (Tensor): Computed loss value
        """
        # Input validation
        assert logits.dim() == 2, "Logits must be 2D tensor"
        assert targets.dim() == 1, "Targets must be 1D tensor"
        assert logits.shape[0] == targets.shape[0], "Batch size mismatch"
        
        # Generate device-agnostic Gaussian noise
        gaussian_noise = torch.randn_like(logits) * self.sigma + self.mu
        
        # Apply dynamic inflation with broadcasting
        inflated_logits = logits + self.delta_base * gaussian_noise
        
        # Compute cross-entropy with stable softmax
        return F.cross_entropy(inflated_logits, targets)

    def extra_repr(self):
        """Display essential parameters"""
        return f"sigma={self.sigma}, mu={self.mu}, num_classes={self.num_classes}"

# if __name__ == '__main__':
#     # Test configuration
#     class_counts = torch.tensor([1000, 500, 100, 50, 10])
    
#     # Initialize loss function
#     loss_fn = GInfLoss(class_counts, sigma=0.1)
#     print("Delta Base Values:", 
#           f"\nExpected: [0.0000, 0.6931, 2.3026, 2.9957, 4.6052]"
#           f"\nActual: {loss_fn.delta_base.detach().numpy().round(4)}")
    
#     # Forward pass verification
#     logits = torch.randn(3, 5)
#     targets = torch.randint(0, 5, (3,))
#     loss = loss_fn(logits, targets)
#     print(f"\nLoss Validation: {loss.item():.4f} (should be positive)")
    
#     # Gradient propagation check
#     logits.requires_grad_(True)
#     loss = loss_fn(logits, targets)
#     loss.backward()
#     print(f"Gradient Check: logits.grad.norm() = {logits.grad.norm().item():.4f} (should be non-zero)")


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
