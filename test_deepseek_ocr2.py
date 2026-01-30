"""
DeepSeek-OCR2 测试脚本 - 带显存优化
"""

import os
import sys
import time

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def check_environment():
    """检查环境"""
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used_mem = torch.cuda.memory_allocated(0) / 1024**3
        free_mem = total_mem - used_mem
        print(f"GPU memory: {total_mem:.1f} GB total, {free_mem:.1f} GB free")
    return torch.cuda.is_available()

def print_gpu_memory(label=""):
    """打印GPU内存使用"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  [GPU {label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def test_deepseek_ocr2(image_path: str = None):
    """测试DeepSeek-OCR2"""
    print("\n" + "=" * 50)
    print("Testing DeepSeek-OCR2")
    print("=" * 50)
    
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    model_name = 'deepseek-ai/DeepSeek-OCR-2'
    
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes...")
    
    start = time.time()
    
    try:
        # 加载tokenizer
        print("\n[Step 1] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("  Tokenizer loaded!")
        print_gpu_memory("after tokenizer")
        
        # 加载模型 - 使用bfloat16节省显存
        print("\n[Step 2] Loading model with bfloat16...")
        print("  (Using bfloat16 to save GPU memory)")
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,  # 使用bfloat16节省显存
            device_map="auto",  # 自动分配设备
            low_cpu_mem_usage=True,  # 降低CPU内存使用
        )
        
        load_time = time.time() - start
        print(f"  Model loaded in {load_time:.1f}s")
        print_gpu_memory("after model load")
        
        # 如果提供了图像路径，进行OCR测试
        if image_path and os.path.exists(image_path):
            print(f"\n[Step 3] Processing image: {image_path}")
            
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            
            print("  Running inference...")
            start = time.time()
            
            try:
                # 设置临时输出目录
                import tempfile
                output_dir = tempfile.mkdtemp()
                
                res = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=output_dir,  # 必须提供有效路径
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=False
                )
                ocr_time = time.time() - start
                
                print(f"  OCR completed in {ocr_time:.2f}s")
                print_gpu_memory("after inference")
                
                print(f"\n[Result]")
                print("-" * 50)
                if isinstance(res, str):
                    print(res[:1000] + "..." if len(res) > 1000 else res)
                else:
                    print(str(res)[:1000])
                print("-" * 50)
                
                return res
                
            except torch.cuda.OutOfMemoryError:
                print("\n!!! GPU OUT OF MEMORY during inference !!!")
                print("Try reducing image_size or using CPU offload")
                return None
                
        else:
            print("\nNo image provided. Model loaded successfully!")
            return None
            
    except torch.cuda.OutOfMemoryError:
        print("\n!!! GPU OUT OF MEMORY !!!")
        print("Your GPU (12GB) may not be enough for this model.")
        print("Options:")
        print("  1. Close other GPU-using applications")
        print("  2. Use 8-bit quantization (requires bitsandbytes)")
        print("  3. Use CPU offload")
        return None
        
    except Exception as e:
        print(f"\n!!! ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_pdf():
    """测试PDF处理"""
    print("\n" + "=" * 50)
    print("Converting PDF to Image")
    print("=" * 50)
    
    # 默认测试PDF
    test_pdf = r"C:\Users\Kevin\Desktop\excel\致同\扫描资料\扫描资料\中区建设\中区西片中小企业\其他文件（根据具体包含的纸质版文件命名）\中区西片法定图制.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"Test PDF not found: {test_pdf}")
        return
    
    # 转换PDF为图像
    print(f"Source: {test_pdf}")
    from pdf2image import convert_from_path
    
    poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"
    images = convert_from_path(test_pdf, first_page=1, last_page=1, dpi=200, poppler_path=poppler_path)
    
    if images:
        # 保存临时图像
        temp_image = "temp_page.png"
        images[0].save(temp_image)
        print(f"Saved: {temp_image} ({images[0].size})")
        
        # 运行OCR
        result = test_deepseek_ocr2(temp_image)
        
        # 清理
        if os.path.exists(temp_image):
            os.remove(temp_image)
        
        return result
    else:
        print("Failed to convert PDF")
        return None

if __name__ == "__main__":
    print("DeepSeek-OCR2 Test Script (Memory Optimized)")
    print("=" * 50)
    
    # 检查环境
    gpu_available = check_environment()
    
    if not gpu_available:
        print("\nWARNING: CUDA not available!")
        sys.exit(1)
    
    # 清理GPU缓存
    import torch
    torch.cuda.empty_cache()
    print("\nGPU cache cleared.")
    
    # 测试
    if len(sys.argv) > 1:
        test_deepseek_ocr2(sys.argv[1])
    else:
        test_with_pdf()
    
    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)
