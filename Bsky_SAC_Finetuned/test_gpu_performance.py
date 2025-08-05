#!/usr/bin/env python3
"""
GPU Performance Test for RTX 5070 Ti
This script helps identify performance bottlenecks in the training setup.
"""

import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np

def test_gpu_setup():
    """Test basic GPU setup and capabilities."""
    print("=== GPU Performance Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Test memory bandwidth
        print("\n=== Memory Bandwidth Test ===")
        device = torch.device("cuda")
        
        # Test different data types
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                print(f"BF16 not supported on this GPU")
                continue
                
            size = 1024 * 1024 * 100  # 100M elements
            data = torch.randn(size, dtype=dtype, device=device)
            
            start_time = time.time()
            for _ in range(10):
                result = data * 2.0
                torch.cuda.synchronize()
            end_time = time.time()
            
            bandwidth = (size * torch.finfo(dtype).bits / 8 * 10) / (end_time - start_time) / 1e9
            print(f"{str(dtype):15}: {bandwidth:.1f} GB/s")
            
            del data, result
            torch.cuda.empty_cache()

def test_model_loading():
    """Test model loading performance."""
    print("\n=== Model Loading Test ===")
    
    model_name = "microsoft/DialoGPT-small"  # Smaller model for testing
    
    # Test different configurations
    configs = [
        {"name": "FP32", "dtype": torch.float32, "quantization": False},
        {"name": "BF16", "dtype": torch.bfloat16, "quantization": False},
        {"name": "4-bit", "dtype": torch.bfloat16, "quantization": True},
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        try:
            start_time = time.time()
            
            if config["quantization"]:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=config["dtype"]
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=config["dtype"]
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=config["dtype"]
                )
            
            load_time = time.time() - start_time
            
            # Test inference speed
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(**inputs)
            
            # Benchmark inference
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(**inputs)
            
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Inference time: {inference_time/10*1000:.1f}ms per forward pass")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  GPU memory: {memory_used:.1f} GB")
            
            del model, tokenizer, inputs
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  Failed: {e}")

def test_training_speed():
    """Test training step performance."""
    print("\n=== Training Speed Test ===")
    
    model_name = "microsoft/DialoGPT-small"
    
    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.train()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    sequence_length = 512
    
    print(f"Testing with sequence length: {sequence_length}")
    
    for batch_size in batch_sizes:
        try:
            # Create dummy batch
            input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, sequence_length)).to(model.device)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            
            # Warmup
            for _ in range(3):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            step_time = (time.time() - start_time) / 10
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            throughput = batch_size / step_time
            
            print(f"  Batch size {batch_size:2d}: {step_time*1000:6.1f}ms/step, {throughput:5.1f} samples/s, {memory_used:.1f}GB")
            
            torch.cuda.reset_peak_memory_stats()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch size {batch_size:2d}: OOM")
        except Exception as e:
            print(f"  Batch size {batch_size:2d}: Error - {e}")
        
        torch.cuda.empty_cache()

def main():
    """Main test function."""
    test_gpu_setup()
    test_model_loading()
    test_training_speed()
    
    print("\n=== Recommendations ===")
    print("Based on the test results:")
    print("1. Use the largest batch size that doesn't cause OOM")
    print("2. Use BF16 if supported for better performance")
    print("3. Consider sequence length vs memory tradeoff")
    print("4. Monitor GPU utilization during training")

if __name__ == "__main__":
    main()
