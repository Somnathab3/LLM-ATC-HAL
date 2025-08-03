#!/usr/bin/env python3
"""
LoRA Size Explosion Explanation
===============================
Demonstrates why a small LoRA adapter creates a large merged model.
"""

import json
import numpy as np
from pathlib import Path

def analyze_lora_size_expansion():
    """Analyze why LoRA merge causes size expansion."""
    
    print("🔍 LoRA Size Expansion Analysis")
    print("=" * 60)
    
    # Read LoRA config
    adapter_config_path = Path("models/llama3.1-bsky-lora/adapter_config.json")
    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
        
        lora_rank = config.get('r', 16)
        lora_alpha = config.get('lora_alpha', 32)
        target_modules = config.get('target_modules', [])
        
        print(f"📊 LoRA Configuration:")
        print(f"   Rank (r): {lora_rank}")
        print(f"   Alpha: {lora_alpha}")
        print(f"   Target modules: {len(target_modules)} modules")
        print(f"   Modules: {target_modules}")
        
    else:
        print("⚠️ Could not read LoRA config, using defaults")
        lora_rank = 16
        target_modules = ['up_proj', 'k_proj', 'o_proj', 'gate_proj', 'v_proj', 'q_proj', 'down_proj']
    
    print(f"\n🧮 Llama 3.1 8B Model Dimensions:")
    
    # Llama 3.1 8B architecture (approximate)
    model_dim = 4096      # Hidden dimension
    intermediate_size = 14336  # FFN intermediate size  
    num_layers = 32       # Number of transformer layers
    num_heads = 32        # Number of attention heads
    head_dim = model_dim // num_heads  # 128
    
    print(f"   Hidden dimension: {model_dim}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Number of layers: {num_layers}")
    print(f"   Attention heads: {num_heads}")
    
    print(f"\n💾 Weight Matrix Sizes (per layer):")
    
    # Calculate original weight sizes
    weights_per_layer = {
        'q_proj': model_dim * model_dim,      # 4096 x 4096
        'k_proj': model_dim * model_dim,      # 4096 x 4096  
        'v_proj': model_dim * model_dim,      # 4096 x 4096
        'o_proj': model_dim * model_dim,      # 4096 x 4096
        'up_proj': model_dim * intermediate_size,   # 4096 x 14336
        'gate_proj': model_dim * intermediate_size, # 4096 x 14336
        'down_proj': intermediate_size * model_dim  # 14336 x 4096
    }
    
    total_params_per_layer = 0
    lora_storage_per_layer = 0
    merged_impact_per_layer = 0
    
    for module, param_count in weights_per_layer.items():
        if module in target_modules:
            # LoRA storage: A matrix (rank x dim1) + B matrix (dim2 x rank)
            if 'proj' in module and module in ['up_proj', 'gate_proj']:
                # For up/gate_proj: 4096 x 14336
                lora_a_params = lora_rank * model_dim        # rank x 4096
                lora_b_params = intermediate_size * lora_rank # 14336 x rank
            elif 'down_proj' in module:
                # For down_proj: 14336 x 4096
                lora_a_params = lora_rank * intermediate_size # rank x 14336
                lora_b_params = model_dim * lora_rank         # 4096 x rank
            else:
                # For attention projections: 4096 x 4096
                lora_a_params = lora_rank * model_dim        # rank x 4096
                lora_b_params = model_dim * lora_rank        # 4096 x rank
            
            lora_params = lora_a_params + lora_b_params
            lora_storage_per_layer += lora_params
            
            # When merged, LoRA affects the ENTIRE original weight matrix
            merged_impact_per_layer += param_count
            
            print(f"   {module}:")
            print(f"     Original: {param_count:,} parameters")
            print(f"     LoRA A+B: {lora_params:,} parameters ({lora_params/param_count*100:.2f}% of original)")
            print(f"     Merged impact: {param_count:,} parameters (100% of original)")
            
        total_params_per_layer += param_count
    
    print(f"\n📏 Per-Layer Summary:")
    print(f"   Total parameters per layer: {total_params_per_layer:,}")
    print(f"   LoRA storage per layer: {lora_storage_per_layer:,}")
    print(f"   LoRA efficiency: {lora_storage_per_layer/total_params_per_layer*100:.2f}%")
    
    print(f"\n🏗️ Full Model Calculations:")
    total_model_params = total_params_per_layer * num_layers
    total_lora_params = lora_storage_per_layer * num_layers
    
    # Convert to approximate file sizes (16-bit precision)
    bytes_per_param = 2  # 16-bit (half precision)
    
    original_model_size_gb = (total_model_params * bytes_per_param) / (1024**3)
    lora_adapter_size_mb = (total_lora_params * bytes_per_param) / (1024**2)
    
    print(f"   Total model parameters: {total_model_params:,}")
    print(f"   Total LoRA parameters: {total_lora_params:,}")
    print(f"   Original model size: ~{original_model_size_gb:.1f} GB")
    print(f"   LoRA adapter size: ~{lora_adapter_size_mb:.1f} MB")
    print(f"   LoRA compression ratio: {total_lora_params/total_model_params*100:.3f}%")
    
    print(f"\n🔄 What Happens During Merge:")
    print(f"   1. Load base model: ~{original_model_size_gb:.1f} GB")
    print(f"   2. Load LoRA adapter: ~{lora_adapter_size_mb:.1f} MB")
    print(f"   3. Compute ΔW = B × A for each matrix")
    print(f"   4. Update: W_new = W_original + ΔW")
    print(f"   5. Save merged model: ~{original_model_size_gb:.1f} GB")
    print(f"   6. Ollama converts to blob format with additional overhead")
    
    print(f"\n💡 Key Insights:")
    print(f"   • LoRA stores tiny matrices that REPRESENT large changes")
    print(f"   • During merge, these expand to full-size weight updates")
    print(f"   • The merged model contains modified versions of ALL original weights")
    print(f"   • File size stays similar to original because we're modifying, not adding")
    print(f"   • Ollama blob format may add overhead for metadata, quantization, etc.")
    
    print(f"\n🎯 Your Case:")
    print(f"   • Base Llama 3.1 8B: 4.58 GB (Ollama blob)")
    print(f"   • LoRA adapter: 160.1 MB")
    print(f"   • Merged model: 14.97 GB (Ollama blob)")
    print(f"   • Size difference suggests Ollama's blob format uses higher precision")
    print(f"     or includes additional metadata/optimization data")

def check_actual_pytorch_model_size():
    """Check the actual PyTorch merged model size."""
    print(f"\n📁 Actual File Sizes:")
    
    merged_path = Path("models/llama3.1-bsky-merged")
    if merged_path.exists():
        total_size = 0
        safetensors_files = list(merged_path.glob("*.safetensors"))
        
        print(f"   PyTorch merged model files:")
        for file in safetensors_files:
            size_gb = file.stat().st_size / (1024**3)
            total_size += file.stat().st_size
            print(f"     {file.name}: {size_gb:.2f} GB")
        
        total_gb = total_size / (1024**3)
        print(f"   Total PyTorch model: {total_gb:.2f} GB")
        print(f"   Ollama blob size: 14.97 GB")
        print(f"   Ollama overhead: {14.97 - total_gb:.2f} GB")
        
        # Check if there's a model index
        index_file = merged_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            print(f"\n📊 Model Sharding Info:")
            print(f"   Number of shards: {len(index_data.get('weight_map', {}))}")
            print(f"   Total parameters: {index_data.get('metadata', {}).get('total_size', 'unknown')}")
    else:
        print(f"   ❌ Merged model directory not found: {merged_path}")

if __name__ == "__main__":
    analyze_lora_size_expansion()
    check_actual_pytorch_model_size()
