import torch 
import torch.nn as nn 
from captum.attr import LRP


def get_prober(prober_type, model_dim, num_layers=1, init_seed=42, **kwargs):
    if prober_type == "BCEProber":
        return BCEProber(model_dim, num_layers, init_seed=init_seed, **kwargs)
    else:
        raise ValueError(f"Prober type {prober_type} not supported")


def make_prober_with_attribution_weight(attribution_weight_dict, layer):
    
    layer = str(layer)
    sorted_indices = attribution_weight_dict[layer]['sorted_indices']
    sorted_values = attribution_weight_dict[layer]['sorted_values']
    hidden_dim = len(sorted_indices)

    original_attribution_weight = torch.zeros(hidden_dim)
    original_attribution_weight[sorted_indices] = torch.tensor(sorted_values)

    value_prober  = get_prober("BCEProber", hidden_dim, num_layers=1, init_seed=42) # seed has no effect on the weight initialization
    def init_with_xavier_like(linear, w, eps=1e-12):
        d = w.numel()
        target_norm = (2 * d / (d + 1)) ** 0.5  # ≈ sqrt(2)
        with torch.no_grad():
            u = w / (w.norm(p=2) + eps) * target_norm
            linear.weight.copy_(u.unsqueeze(0).repeat(linear.weight.size(0), 1))
            linear.bias.zero_()
            
    init_with_xavier_like(value_prober.model, original_attribution_weight)
    return value_prober


class BCEProber(nn.Module):
    def __init__(self, model_dim, num_layers=1, hidden_dim=256, bias=True, init_seed=42, **kwargs):
        super().__init__()
        if num_layers ==1:
            self.model = nn.Linear(model_dim, 1, bias=bias)
        elif num_layers == 2:
            self.model = nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1, bias=bias)
            )
        else:
            raise ValueError(f"Number of layers {num_layers} not supported")
        self.init_seed = init_seed
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(self.init_seed)
        if hasattr(self.model, 'weight') and self.model.weight is not None:
            nn.init.xavier_uniform_(self.model.weight)
        if hasattr(self.model, 'bias') and self.model.bias is not None:
            nn.init.zeros_(self.model.bias)
            
    def forward(self, x):
        if x.ndim == 3:
            x = x[:, -1, :]
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs  # or return logits, probs if you want both

    def compute_loss(self, logits, y):
        y = y.float().unsqueeze(1)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        return loss
    
    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            probs = self.forward(x).squeeze(1)
            preds = (probs > threshold).long()
        return preds, probs

    def lrp(self, x):
        is_training = self.training
        self.eval()
        
        # only a single target is supported for BCEProber
        target = 0  
    
        lrp = LRP(self)
        if isinstance(target, int):
            n_targets = x.shape[0]
            target = torch.tensor([target] * n_targets).to(x.device)
        attributions = lrp.attribute(x, target)
        attributions = attributions.abs()
        sums = attributions.sum(dim=1).unsqueeze(1)
        # If sum is 0, distribute equally
        mask = (sums == 0)
        attributions = torch.where(mask, 
                                 torch.ones_like(attributions) / attributions.shape[1],
                                 attributions / sums)
        assert torch.all(torch.abs(attributions.sum(dim=1) - 1) < 1e-6)
        
        self.train(is_training)
        return attributions