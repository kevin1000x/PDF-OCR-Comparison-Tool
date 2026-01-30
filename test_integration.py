"""
快速测试DeepSeek-OCR2集成
"""

import os
import sys
import yaml

# 确保在deepseek-ocr2环境中运行
print("Testing DeepSeek-OCR2 Integration")
print("=" * 50)

# 加载配置
config_path = "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print(f"OCR engine in config: {config['ocr'].get('engine', 'paddleocr')}")

# 导入PDF处理器
from pdf_processor import PDFProcessor

# 导入OCR引擎
from main import get_ocr_engine

print("\nInitializing OCR engine...")
ocr_engine = get_ocr_engine(config.get('ocr', {}))
print(f"Engine type: {type(ocr_engine).__name__}")

# 测试PDF
test_pdf = r"C:\Users\Kevin\Desktop\excel\致同\扫描资料\扫描资料\中区建设\中区西片中小企业\其他文件（根据具体包含的纸质版文件命名）\中区西片法定图制.pdf"

if not os.path.exists(test_pdf):
    print(f"Test PDF not found: {test_pdf}")
    sys.exit(1)

print(f"\nTest PDF: {test_pdf}")

# 初始化PDF处理器
pdf_processor = PDFProcessor(config)

# 处理第一页
print("\nProcessing first page...")
import time

for page_num, image in pdf_processor.process_pdf(test_pdf):
    if page_num > 1:
        break
    
    print(f"  Page {page_num}, image size: {image.shape if hasattr(image, 'shape') else 'N/A'}")
    
    start = time.time()
    result = ocr_engine.recognize_pdf_page(image, page_num)
    elapsed = time.time() - start
    
    print(f"  OCR completed in {elapsed:.2f}s")
    print(f"  Results count: {len(result.results)}")
    
    # 显示文本
    text = result.get_full_text()
    print(f"\n[Recognized Text]")
    print("-" * 50)
    print(text[:500] if len(text) > 500 else text)
    print("-" * 50)

print("\n✅ Integration test complete!")
