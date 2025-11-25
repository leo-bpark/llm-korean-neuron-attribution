"""
FastAPI server for LLM neuron attribution visualization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

from LKN.attribution import NeuronAttribution

app = FastAPI(title="LLM Neuron Attribution Tool")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache: Dict[str, tuple] = {}  # model_name -> (model, tokenizer, attribution_calculator)

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")


class LoadModelRequest(BaseModel):
    model_name: str


class AttributionRequest(BaseModel):
    model_name: str
    input_text: str
    layer_idx: int
    neuron_idx: int
    method: str = "activation_patching"  # "activation_patching" or "integrated_gradients"


class NeuronInfoRequest(BaseModel):
    model_name: str
    layer_idx: int


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Index page not found")


@app.post("/api/load_model")
async def load_model(request: LoadModelRequest):
    """Load a model and tokenizer."""
    try:
        if request.model_name in model_cache:
            model, tokenizer, _ = model_cache[request.model_name]
            return {
                "status": "success",
                "message": f"Model {request.model_name} already loaded",
                "model_name": request.model_name
            }
        
        print(f"Loading model: {request.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            request.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create attribution calculator
        attribution_calc = NeuronAttribution(model, tokenizer)
        
        # Cache the model
        model_cache[request.model_name] = (model, tokenizer, attribution_calc)
        
        # Get model info
        num_layers = len(model.transformer.h) if hasattr(model, 'transformer') else \
                    len(model.gpt_neox.layers) if hasattr(model, 'gpt_neox') else \
                    len(model.model.layers) if hasattr(model, 'model') else 0
        
        return {
            "status": "success",
            "message": f"Model {request.model_name} loaded successfully",
            "model_name": request.model_name,
            "num_layers": num_layers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/api/get_neuron_info")
async def get_neuron_info(request: NeuronInfoRequest):
    """Get information about neurons in a specific layer."""
    try:
        if request.model_name not in model_cache:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        model, tokenizer, _ = model_cache[request.model_name]
        
        # Find layer
        layer = model.transformer.h[request.layer_idx] if hasattr(model, 'transformer') else \
                model.gpt_neox.layers[request.layer_idx] if hasattr(model, 'gpt_neox') else \
                model.model.layers[request.layer_idx] if hasattr(model, 'model') else None
        
        if layer is None:
            raise HTTPException(status_code=404, detail=f"Layer {request.layer_idx} not found")
        
        # Find MLP module and get hidden dimension
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_fc'):
            # GPT-2 style
            hidden_dim = layer.mlp.c_fc.out_features
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'dense_h_to_4h'):
            # GPT-NeoX style
            hidden_dim = layer.mlp.dense_h_to_4h.out_features
        elif hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'gate_proj'):
            # LLaMA style
            hidden_dim = layer.feed_forward.gate_proj.out_features
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'fc_in'):
            hidden_dim = layer.mlp.fc_in.out_features
        else:
            # Try to infer from model config
            if hasattr(model, 'config'):
                hidden_dim = getattr(model.config, 'intermediate_size', 
                                   getattr(model.config, 'ffn_dim', 4096))
            else:
                hidden_dim = 4096  # Default guess
        
        return {
            "layer_idx": request.layer_idx,
            "num_neurons": hidden_dim,
            "neuron_indices": list(range(hidden_dim))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting neuron info: {str(e)}")


@app.post("/api/compute_attribution")
async def compute_attribution(request: AttributionRequest):
    """Compute attribution for a specific neuron."""
    try:
        if request.model_name not in model_cache:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        model, tokenizer, attribution_calc = model_cache[request.model_name]
        
        # Tokenize input
        inputs = tokenizer(request.input_text, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"]
        
        # Get tokens for visualization
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Compute attribution
        if request.method == "integrated_gradients":
            attribution_embeds = attribution_calc.compute_integrated_gradients(
                input_ids=input_ids,
                layer_idx=request.layer_idx,
                neuron_idx=request.neuron_idx
            )
        else:  # activation_patching
            attribution_embeds = attribution_calc.compute_activation_patching(
                input_ids=input_ids,
                layer_idx=request.layer_idx,
                neuron_idx=request.neuron_idx
            )
        
        # Propagate to token level (abs mean)
        token_attributions = attribution_calc.propagate_to_tokens(
            attribution_embeds,
            method="abs_mean"
        )
        
        # Normalize attributions for visualization (0-1 scale)
        if token_attributions.max() > token_attributions.min():
            normalized_attributions = (token_attributions - token_attributions.min()) / \
                                     (token_attributions.max() - token_attributions.min())
        else:
            normalized_attributions = token_attributions
        
        # Prepare response
        result = {
            "tokens": tokens,
            "attributions": normalized_attributions.tolist(),
            "raw_attributions": token_attributions.tolist(),
            "layer_idx": request.layer_idx,
            "neuron_idx": request.neuron_idx
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error computing attribution: {str(e)}")


@app.get("/api/list_models")
async def list_models():
    """List currently loaded models."""
    return {
        "models": list(model_cache.keys())
    }


if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # Mount static files
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

