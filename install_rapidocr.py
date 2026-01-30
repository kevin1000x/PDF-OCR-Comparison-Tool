"""
RapidOCR安装和测试脚本
RapidOCR使用PaddleOCR的模型但通过ONNX运行，精度相同但更快
"""

import subprocess
import sys

print("=" * 50)
print("Installing RapidOCR")
print("=" * 50)

# 安装RapidOCR
print("\n1. Installing rapidocr-onnxruntime...")
subprocess.run([sys.executable, "-m", "pip", "install", "rapidocr-onnxruntime", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

# 尝试安装GPU版本
print("\n2. Installing onnxruntime-gpu for CUDA acceleration...")
subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

print("\n" + "=" * 50)
print("Testing RapidOCR")
print("=" * 50)

import time

# 测试RapidOCR
print("\n3. Testing RapidOCR...")
try:
    from rapidocr_onnxruntime import RapidOCR
    import numpy as np
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    print("   Initializing RapidOCR...")
    start = time.time()
    ocr = RapidOCR()
    print(f"   Init time: {time.time() - start:.2f}s")
    
    print("   Running OCR on test image...")
    start = time.time()
    result, elapse = ocr(test_img)
    print(f"   OCR time: {time.time() - start:.2f}s")
    print(f"   Internal elapse: {elapse}")
    
    print("\n✓ RapidOCR is working!")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# 测试真实PDF
print("\n4. Testing with real PDF...")
try:
    from rapidocr_onnxruntime import RapidOCR
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    
    test_pdf = r"C:\Users\Kevin\Desktop\excel\致同\扫描资料\扫描资料\中区建设\中区西片中小企业\其他文件（根据具体包含的纸质版文件命名）\中区西片法定图制.pdf"
    
    print(f"   Loading PDF: {test_pdf}")
    images = convert_from_path(test_pdf, first_page=1, last_page=1, dpi=200, 
                               poppler_path=r"C:\poppler\poppler-25.12.0\Library\bin")
    
    if images:
        img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        print(f"   Image size: {img.shape}")
        
        print("   Running OCR...")
        ocr = RapidOCR()
        start = time.time()
        result, elapse = ocr(img)
        ocr_time = time.time() - start
        
        print(f"   OCR time: {ocr_time:.2f}s")
        
        if result:
            text = " ".join([line[1] for line in result])
            print(f"   Text: {text[:200]}...")
            print(f"\n✓ Success! Single page processed in {ocr_time:.2f}s")
        else:
            print("   No text found")
            
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("If RapidOCR works faster, we can update the OCR engine!")
print("=" * 50)
