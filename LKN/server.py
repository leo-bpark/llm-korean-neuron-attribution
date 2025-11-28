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
from LKN.utils import decode_and_merge_tokens

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
    neurons: List[List[int]]  # List of [layer_idx, neuron_idx] lists, e.g., [[27, 10101], [5, 10]]


class NeuronInfoRequest(BaseModel):
    model_name: str
    layer_idx: int


class EnvConfigRequest(BaseModel):
    """Environment / runtime configuration that can be tweaked from the UI.

    NOTE:
    - CUDA 관련 환경변수(CUDA_VISIBLE_DEVICES)는 **모델 로드 전에** 설정될 때만
      제대로 반영됩니다.
    - 이미 로드된 모델에는 영향을 주지 않을 수 있습니다.
    """
    cuda_visible_devices: Optional[str] = None


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Index page not found")


@app.post("/api/set_env")
async def set_env(req: EnvConfigRequest):
    """Set simple environment options such as CUDA_VISIBLE_DEVICES.

    FastAPI 프로세스의 os.environ 에 값을 기록해 두고,
    이후 로드되는 모델들이 이 설정을 보도록 합니다.
    """
    try:
        updated: Dict[str, str] = {}

        if req.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = req.cuda_visible_devices
            updated["CUDA_VISIBLE_DEVICES"] = req.cuda_visible_devices

        # 안내 메시지: 이미 로드된 모델이 있으면 효과가 제한적일 수 있음
        has_loaded_model = len(model_cache) > 0

        return {
            "status": "success",
            "updated": updated,
            "has_loaded_model": has_loaded_model,
            "message": (
                "환경 설정이 저장되었습니다. 이미 로드된 모델에는 바로 적용되지 않을 수 있습니다."
                if has_loaded_model
                else "환경 설정이 저장되었습니다. 이후 로드되는 모델에 적용됩니다."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting environment: {str(e)}")


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
    """Compute attribution for multiple neurons using the IG method from LKN.ig."""
    try:
        if request.model_name not in model_cache:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        # model_cache 에는 (model, tokenizer, attribution_calc) 가 들어있을 수 있으므로
        cached = model_cache[request.model_name]
        if isinstance(cached, tuple) and len(cached) >= 2:
            model, tokenizer = cached[0], cached[1]
        else:
            raise HTTPException(status_code=500, detail="Invalid model cache entry")
        
        # Convert neurons list to list of tuples
        neurons = [tuple(n) for n in request.neurons]  # e.g., [(27, 10101), (5, 10)]
        
        # Use the attribute function from LKN.ig (same as notebook)
        from LKN.ig import attribute
        result = attribute(model, tokenizer, request.model_name, request.input_text, neurons)
        
        # result is list of (token_id, attr) tuples
        # Use decode_and_merge_tokens to merge Korean tokens properly
        token_list = [token_id for token_id, _ in result]
        attr_list = [float(attr.item() if hasattr(attr, 'item') else attr) for _, attr in result]
        merged_tokens, merged_attributions = decode_and_merge_tokens(token_list, attr_list, tokenizer)

        # Normalize attributions for visualization (0-1 scale)
        import numpy as np
        arr = np.array(merged_attributions, dtype=float)
        if arr.size > 0 and arr.max() > arr.min():
            normalized = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            normalized = arr
        
        # Prepare response
        return {
            "tokens": merged_tokens,
            "attributions": normalized.tolist(),
            "raw_attributions": merged_attributions,
            "neurons": request.neurons
        }
        
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


class GetBiasConceptsRequest(BaseModel):
    model_name: str


class GetTopKNeuronsRequest(BaseModel):
    model_name: str
    concept: str


@app.post("/api/get_bias_concepts")
async def get_bias_concepts(request: GetBiasConceptsRequest):
    """Get list of bias concepts from top_k_neuron_results.json."""
    try:
        import json
        
        # Construct file path
        model_path = request.model_name.replace("/", "-")
        file_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "model_info",
            request.model_name,
            "top_k_neuron_results.json"
        )
        
        if not os.path.exists(file_path):
            return {
                "concepts": [],
                "message": f"File not found: {file_path}"
            }
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        concepts = list(data.keys())
        
        return {
            "concepts": concepts,
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading concepts: {str(e)}")


@app.post("/api/get_topk_neurons")
async def get_topk_neurons(request: GetTopKNeuronsRequest):
    """Get topK neurons for a specific concept."""
    try:
        import json
        
        # Construct file path
        file_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "model_info",
            request.model_name,
            "top_k_neuron_results.json"
        )
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if request.concept not in data:
            raise HTTPException(status_code=404, detail=f"Concept '{request.concept}' not found")
        
        neurons = data[request.concept]
        
        return {
            "concept": request.concept,
            "neurons": neurons,
            "model_name": request.model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading topK neurons: {str(e)}")


class GetConceptExamplesRequest(BaseModel):
    concept: str


@app.post("/api/get_concept_examples")
async def get_concept_examples(request: GetConceptExamplesRequest):
    """Get pos/neg examples for a specific concept from sample_2.json."""
    try:
        import json
        
        # Construct file path
        file_path = os.path.join(PROJECT_ROOT, "data", "sample_2.json")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if request.concept not in data:
            raise HTTPException(status_code=404, detail=f"Concept '{request.concept}' not found")
        
        concept_data = data[request.concept]
        
        return {
            "concept": request.concept,
            "pos": concept_data.get("pos", []),
            "neg": concept_data.get("neg", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading concept examples: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # Mount static files
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

