"""
GPU使用诊断脚本
"""
import os
import sys

print("=" * 60)
print("GPU Diagnostic")
print("=" * 60)

# 1. 检查PyTorch CUDA
print("\n[1] PyTorch CUDA Status:")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device count: {torch.cuda.device_count()}")
        
        # 显存信息
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  Total memory: {total:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    else:
        print("  ⚠️ CUDA NOT AVAILABLE!")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. 检查nvidia-smi
print("\n[2] NVIDIA-SMI Output:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', 
                            '--format=csv,noheader,nounits'], 
                           capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        parts = result.stdout.strip().split(',')
        print(f"  GPU: {parts[0].strip()}")
        print(f"  Memory used: {parts[1].strip()} MB / {parts[2].strip()} MB")
        print(f"  GPU utilization: {parts[3].strip()}%")
    else:
        print(f"  Error: {result.stderr}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. 检查模型设备
print("\n[3] DeepSeek-OCR2 Model Device Check:")
try:
    from transformers import AutoModel, AutoTokenizer
    
    model_name = 'deepseek-ai/DeepSeek-OCR-2'
    print(f"  Loading model (this may take a moment)...")
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # 检查模型参数在哪个设备
    for name, param in model.named_parameters():
        print(f"  First param '{name[:50]}...' on device: {param.device}")
        break
    
    # 打印device_map
    if hasattr(model, 'hf_device_map'):
        print(f"  Device map: {dict(list(model.hf_device_map.items())[:3])}...")
    
    print("\n  ✅ Model loaded on GPU" if 'cuda' in str(param.device) else "\n  ⚠️ Model NOT on GPU!")
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# 4. 再次检查GPU内存
print("\n[4] GPU Memory After Model Load:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")

print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)
