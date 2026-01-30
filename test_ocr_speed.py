"""
OCR速度测试脚本
测试单个PDF页面的OCR处理时间
"""

import time
import sys
from pathlib import Path

def test_ocr_speed(pdf_path: str):
    """测试OCR速度"""
    import yaml
    
    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing OCR on: {pdf_path}")
    print(f"DPI: {config.get('processing', {}).get('dpi', 200)}")
    print()
    
    # 初始化组件
    print("Step 1: Initializing PDF processor...")
    start = time.time()
    
    from pdf_processor import PDFToImage, PDFReader
    
    pdf_to_image = PDFToImage(dpi=config.get('processing', {}).get('dpi', 150))
    reader = PDFReader()
    
    print(f"  PDF processor ready: {time.time() - start:.1f}s")
    
    # 获取页数
    page_count = reader.get_page_count(pdf_path)
    print(f"  Total pages: {page_count}")
    print()
    
    # 转换第一页
    print("Step 2: Converting PDF page to image...")
    convert_start = time.time()
    image = pdf_to_image.convert_page(pdf_path, 1)
    convert_time = time.time() - convert_start
    print(f"  Convert time: {convert_time:.2f}s")
    
    if image:
        import numpy as np
        from PIL import Image as PILImage
        import cv2
        
        # 转换图像格式
        if isinstance(image, PILImage.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image
            
        print(f"  Image size: {img_array.shape}")
        print()
        
        # 初始化OCR
        print("Step 3: Initializing OCR engine...")
        ocr_init_start = time.time()
        from ocr_engine import OCREngine
        ocr_engine = OCREngine(config.get('ocr', {}))
        print(f"  OCR engine created: {time.time() - ocr_init_start:.1f}s")
        
        # OCR识别
        print("Step 4: Running OCR (this may take a while first time)...")
        ocr_start = time.time()
        
        # 打印进度点
        import threading
        stop_dots = False
        def print_dots():
            count = 0
            while not stop_dots:
                count += 1
                if count % 5 == 0:
                    print(f"  ... still running ({count}s)", flush=True)
                time.sleep(1)
        
        dot_thread = threading.Thread(target=print_dots, daemon=True)
        dot_thread.start()
        
        results = ocr_engine.recognize_image(img_array)
        
        stop_dots = True
        ocr_time = time.time() - ocr_start
        print(f"  OCR time: {ocr_time:.2f}s")
        print(f"  Results: {len(results)} text blocks")
        
        # 显示部分文本
        if results:
            all_text = " ".join([r.text for r in results])
            print(f"  Text preview: {all_text[:200]}...")
        else:
            print("  WARNING: No text recognized!")
    
    print()
    print("="*50)
    total_time = convert_time + ocr_time if image else 0
    print(f"Total page processing time: {total_time:.2f}s")
    if page_count > 0 and total_time > 0:
        print(f"Estimated time for {page_count} pages: {total_time * page_count:.1f}s ({total_time * page_count / 60:.1f} min)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认测试文件
        test_pdf = r"C:\Users\Kevin\Desktop\excel\致同\扫描资料\扫描资料\中区建设\中区西片中小企业\其他文件（根据具体包含的纸质版文件命名）\中区西片法定图制.pdf"
    else:
        test_pdf = sys.argv[1]
    
    test_ocr_speed(test_pdf)
