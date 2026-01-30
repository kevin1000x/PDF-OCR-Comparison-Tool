"""
快速测试DeepSeek-OCR2修复版
"""

import os
import sys
import time

print("Testing DeepSeek-OCR2 Fix")
print("=" * 50)

# 测试PDF
test_pdf = r"C:\Users\Kevin\Desktop\excel\致同\扫描资料\扫描资料\中区建设\中区西片中小企业\其他文件（根据具体包含的纸质版文件命名）\中区西片法定图制.pdf"

if not os.path.exists(test_pdf):
    print(f"Test PDF not found: {test_pdf}")
    sys.exit(1)

# 转换PDF为图像
print("\n1. Converting PDF to image...")
from pdf2image import convert_from_path

poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"
images = convert_from_path(test_pdf, first_page=1, last_page=1, dpi=200, poppler_path=poppler_path)

if not images:
    print("Failed to convert PDF")
    sys.exit(1)

temp_image = "temp_test.png"
images[0].save(temp_image)
print(f"   Saved: {temp_image}")

# 测试OCR引擎
print("\n2. Testing DeepSeekOCR2Engine...")
from deepseek_ocr2_engine import DeepSeekOCR2Engine

engine = DeepSeekOCR2Engine({})

print("\n3. Running OCR...")
start = time.time()
result = engine.recognize_pdf_page(temp_image, 1)
elapsed = time.time() - start

print(f"\n4. Results ({elapsed:.2f}s):")
print("-" * 50)
print(f"   Results count: {len(result.results)}")
print(f"   Full text length: {len(result.get_full_text())} chars")
print("\n   Text preview:")
text = result.get_full_text()
print(text[:500] if len(text) > 500 else text if text else "(empty)")
print("-" * 50)

# 清理
if os.path.exists(temp_image):
    os.remove(temp_image)

if len(result.results) > 0:
    print("\n✅ DeepSeek-OCR2 fix verified!")
else:
    print("\n❌ Still returning empty results - need further investigation")
