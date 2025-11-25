"""
MLP neuron attribution calculation module.
Computes attribution scores for MLP neurons and propagates them to token embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np


class NeuronAttribution:
    """Calculate attribution scores for MLP neurons."""
    
    def __init__(self, model, tokenizer, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store hooks for intermediate activations
        self.activation_cache = {}
        self.hooks = []
        
    def _register_hooks(self, layer_idx: int, neuron_idx: int):
        """Register forward hooks to capture MLP activations."""
        self.activation_cache.clear()
        
        def get_activation(name):
            def hook(module, input, output):
                # Store activation for the specific neuron
                if isinstance(output, tuple):
                    output = output[0]
                # output shape: [batch, seq_len, hidden_dim]
                self.activation_cache[name] = output
            return hook
        
        # Find the layer
        layer = None
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Could not find transformer layers in model")
        
        if layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} not found. Model has {len(layers)} layers.")
        
        layer = layers[layer_idx]
        
        # Find MLP/feed-forward module and hook point
        mlp_module = None
        hook_module = None
        
        if hasattr(layer, 'mlp'):
            mlp_module = layer.mlp
            # Hook on the final output of MLP (after activation and projection)
            if hasattr(mlp_module, 'c_proj'):  # GPT-2 style: mlp.c_proj
                hook_module = mlp_module.c_proj
            elif hasattr(mlp_module, 'dense_4h_to_h'):  # GPT-NeoX style
                hook_module = mlp_module.dense_4h_to_h
            elif hasattr(mlp_module, 'out_proj'):  # Some models
                hook_module = mlp_module.out_proj
            else:
                hook_module = mlp_module  # Hook on whole MLP
        elif hasattr(layer, 'feed_forward'):
            mlp_module = layer.feed_forward
            # LLaMA style: feed_forward has up_proj, gate_proj, down_proj
            if hasattr(mlp_module, 'down_proj'):
                hook_module = mlp_module.down_proj
            else:
                hook_module = mlp_module
        elif hasattr(layer, 'c_fc'):  # Direct GPT-2 style layer
            mlp_module = layer
            hook_module = layer.c_proj if hasattr(layer, 'c_proj') else layer
        
        if mlp_module is None or hook_module is None:
            raise ValueError(f"Could not find MLP module in layer {layer_idx}")
        
        # Register hook on MLP output
        hook_handle = hook_module.register_forward_hook(get_activation(f"layer_{layer_idx}_mlp"))
        self.hooks.append(hook_handle)
        
        return mlp_module
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def compute_integrated_gradients(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        neuron_idx: int,
        num_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute integrated gradients for a specific neuron.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            layer_idx: Layer index containing the MLP
            neuron_idx: Neuron index in the MLP
            num_steps: Number of integration steps
            baseline: Baseline input (default: zero embedding)
        
        Returns:
            Attribution scores for each token position [seq_len, hidden_dim]
        """
        input_ids = input_ids.to(self.device)
        
        # Create baseline (zero embeddings)
        if baseline is None:
            baseline_ids = torch.zeros_like(input_ids)
        else:
            baseline_ids = baseline.to(self.device)
        
        # Register hooks
        mlp_module = self._register_hooks(layer_idx, neuron_idx)
        
        try:
            # Get embeddings
            if hasattr(self.model, 'transformer'):
                embed_module = self.model.transformer.wte
            elif hasattr(self.model, 'gpt_neox'):
                embed_module = self.model.gpt_neox.embed_in
            elif hasattr(self.model, 'model'):
                embed_module = self.model.model.embed_tokens
            else:
                embed_module = self.model.get_input_embeddings()
            
            # Compute integrated gradients
            alphas = torch.linspace(0, 1, num_steps + 1, device=self.device)
            
            integrated_grads = None
            
            for alpha in alphas[1:]:  # Skip alpha=0
                # Interpolated input
                baseline_embeds = embed_module(baseline_ids)
                input_embeds = embed_module(input_ids)
                interpolated_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
                interpolated_embeds.requires_grad_(True)
                
                # Forward pass
                self.activation_cache.clear()
                outputs = self.model(inputs_embeds=interpolated_embeds)
                
                # Get neuron activation
                activation_key = f"layer_{layer_idx}_mlp"
                if activation_key not in self.activation_cache:
                    raise ValueError(f"Activation not found for layer {layer_idx}")
                
                neuron_activation = self.activation_cache[activation_key][:, :, neuron_idx]
                
                # Sum over sequence and batch for attribution
                neuron_score = neuron_activation.sum()
                
                # Backward pass
                gradients = torch.autograd.grad(
                    neuron_score,
                    interpolated_embeds,
                    create_graph=False,
                    retain_graph=True
                )[0]
                
                if integrated_grads is None:
                    integrated_grads = gradients.detach()
                else:
                    integrated_grads += gradients.detach()
            
            # Average and multiply by (input - baseline)
            integrated_grads = integrated_grads / num_steps
            input_embeds_final = embed_module(input_ids).detach()
            baseline_embeds_final = embed_module(baseline_ids).detach()
            
            attribution = integrated_grads * (input_embeds_final - baseline_embeds_final)
            
            # Return attribution for first batch item
            return attribution[0].cpu()  # [seq_len, hidden_dim]
            
        finally:
            self._remove_hooks()
    
    def compute_activation_patching(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        neuron_idx: int,
        reference_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attribution using gradient-based method (simpler and faster).
        Uses gradients of neuron activation w.r.t. input embeddings.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            layer_idx: Layer index
            neuron_idx: Neuron index
            reference_input_ids: Not used, kept for API compatibility
        
        Returns:
            Attribution scores propagated to embeddings [seq_len, hidden_dim]
        """
        input_ids = input_ids.to(self.device)
        
        # Register hooks
        mlp_module = self._register_hooks(layer_idx, neuron_idx)
        
        try:
            # Get embedding module
            embed_module = None
            if hasattr(self.model, 'get_input_embeddings'):
                embed_module = self.model.get_input_embeddings()
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                embed_module = self.model.transformer.wte
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_module = self.model.model.embed_tokens
            elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'embed_in'):
                embed_module = self.model.gpt_neox.embed_in
            
            if embed_module is None:
                raise ValueError("Could not find embedding module")
            
            # Get input embeddings
            input_embeds = embed_module(input_ids)
            input_embeds.requires_grad_(True)
            
            # Forward pass
            self.activation_cache.clear()
            outputs = self.model(inputs_embeds=input_embeds)
            
            # Get neuron activation
            activation_key = f"layer_{layer_idx}_mlp"
            if activation_key not in self.activation_cache:
                raise ValueError(f"Activation not found for layer {layer_idx}")
            
            # Get activation for the specific neuron
            mlp_output = self.activation_cache[activation_key]  # [batch, seq_len, hidden_dim]
            neuron_activation = mlp_output[:, :, neuron_idx]  # [batch, seq_len]
            
            # Sum over sequence to get a scalar score
            neuron_score = neuron_activation.sum()
            
            # Backward pass to get gradients
            gradients = torch.autograd.grad(
                neuron_score,
                input_embeds,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Use gradient * input as attribution (gradient * input method)
            attribution = gradients.detach() * input_embeds.detach()
            
            return attribution[0].cpu()  # [seq_len, hidden_dim]
            
        finally:
            self._remove_hooks()
    
    def propagate_to_tokens(
        self,
        attribution_embeds: torch.Tensor,
        method: str = "abs_mean"
    ) -> np.ndarray:
        """
        Propagate embedding attribution to token-level scores.
        
        Args:
            attribution_embeds: Attribution scores per embedding [seq_len, hidden_dim]
            method: Aggregation method ("abs_mean", "mean", "norm")
        
        Returns:
            Token-level attribution scores [seq_len]
        """
        if method == "abs_mean":
            return attribution_embeds.abs().mean(dim=-1).numpy()
        elif method == "mean":
            return attribution_embeds.mean(dim=-1).numpy()
        elif method == "norm":
            return torch.norm(attribution_embeds, dim=-1).numpy()
        else:
            raise ValueError(f"Unknown method: {method}")


