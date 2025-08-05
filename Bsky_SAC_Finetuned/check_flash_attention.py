#!/usr/bin/env python3
"""
Check if Flash Attention is available and working
"""

import torch
import sys

def check_flash_attention():
    """Check Flash Attention availability."""
    print("=== Flash Attention Check ===")
    
    try:
        import flash_attn
        print(f"✅ Flash Attention installed: version {flash_attn.__version__}")
        
        # Check if it works with current setup
        try:
            from flash_attn import flash_attn_func
            print("✅ Flash Attention functions available")
            
            # Test if it works
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                q = torch.randn(1, 8, 64, 64, device=device, dtype=torch.bfloat16)
                k = torch.randn(1, 8, 64, 64, device=device, dtype=torch.bfloat16)
                v = torch.randn(1, 8, 64, 64, device=device, dtype=torch.bfloat16)
                
                try:
                    out = flash_attn_func(q, k, v)
                    print("✅ Flash Attention working correctly")
                    return True
                except Exception as e:
                    print(f"❌ Flash Attention test failed: {e}")
                    return False
            else:
                print("⚠️  CUDA not available, cannot test Flash Attention")
                return False
                
        except ImportError as e:
            print(f"❌ Flash Attention functions not available: {e}")
            return False
            
    except ImportError:
        print("❌ Flash Attention not installed")
        print("Install with: pip install flash-attn --no-build-isolation")
        return False

def check_transformers_version():
    """Check if transformers supports flash attention."""
    try:
        import transformers
        print(f"\n=== Transformers Version ===")
        print(f"Transformers version: {transformers.__version__}")
        
        # Check if flash attention is supported
        from packaging import version
        min_version = version.parse("4.36.0")
        current_version = version.parse(transformers.__version__)
        
        if current_version >= min_version:
            print("✅ Transformers version supports Flash Attention")
            return True
        else:
            print(f"❌ Transformers version too old. Need >= 4.36.0, have {transformers.__version__}")
            print("Upgrade with: pip install transformers>=4.36.0")
            return False
            
    except ImportError:
        print("❌ Transformers not installed")
        return False

def check_torch_version():
    """Check PyTorch version compatibility."""
    print(f"\n=== PyTorch Version ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        # Check if BF16 is supported
        if torch.cuda.is_bf16_supported():
            print("✅ BF16 supported")
        else:
            print("⚠️  BF16 not supported on this GPU")
            
        return True
    else:
        print("❌ CUDA not available")
        return False

def main():
    print("GPU and Flash Attention Compatibility Check")
    print("=" * 50)
    
    torch_ok = check_torch_version()
    transformers_ok = check_transformers_version()
    flash_ok = check_flash_attention()
    
    print("\n=== Summary ===")
    if torch_ok and transformers_ok and flash_ok:
        print("✅ All components ready for optimized training!")
        print("\nRecommended model settings:")
        print("- attn_implementation='flash_attention_2'")
        print("- torch_dtype=torch.bfloat16")
        print("- use_cache=False (during training)")
    elif torch_ok and transformers_ok:
        print("⚠️  Basic setup ready, but Flash Attention not available")
        print("Training will work but may be slower")
        print("\nFallback model settings:")
        print("- attn_implementation='eager'") 
        print("- torch_dtype=torch.bfloat16")
    else:
        print("❌ Setup incomplete. Check the issues above.")
        
    print(f"\nFor RTX 5070 Ti optimization:")
    print("- Batch size: 8-16")
    print("- Sequence length: 1024-2048")
    print("- Gradient accumulation: 1-2")
    print("- Use bf16 precision")

if __name__ == "__main__":
    main()
