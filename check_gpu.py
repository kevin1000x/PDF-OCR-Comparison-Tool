"""
检查GPU状态和PaddleOCR是否使用GPU
"""

print("=" * 50)
print("GPU Detection Test")
print("=" * 50)

# 检查CUDA
print("\n1. Checking CUDA availability...")
try:
    import paddle
    print(f"   PaddlePaddle version: {paddle.__version__}")
    print(f"   CUDA available: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        print(f"   GPU count: {paddle.device.cuda.device_count()}")
        print(f"   Current device: {paddle.device.get_device()}")
except Exception as e:
    print(f"   Error: {e}")

# 检查nvidia-smi
print("\n2. Checking NVIDIA GPU...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   Found GPU: {result.stdout.strip()}")
    else:
        print(f"   nvidia-smi failed: {result.stderr}")
except FileNotFoundError:
    print("   nvidia-smi not found")
except Exception as e:
    print(f"   Error: {e}")

# 测试PaddleOCR GPU使用
print("\n3. Testing PaddleOCR GPU usage...")
import time
try:
    from paddleocr import PaddleOCR
    import numpy as np
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    print("   Initializing PaddleOCR...")
    start = time.time()
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print(f"   Init time: {time.time() - start:.1f}s")
    
    print("   Running OCR on test image...")
    start = time.time()
    result = ocr.predict(input=test_img)
    print(f"   OCR time: {time.time() - start:.1f}s")
    
    # 检查是否在GPU上运行
    import paddle
    device = paddle.device.get_device()
    print(f"   Device used: {device}")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("If GPU is available but device shows 'cpu',")
print("we need to enable GPU mode for PaddleOCR")
print("=" * 50)
