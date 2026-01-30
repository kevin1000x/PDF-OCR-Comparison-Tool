"""
GPU加速性能测试脚本
=====================

用于测试DeepSeek-OCR2在不同配置下的性能
"""

import os
import sys
import time
from pathlib import Path

# 测试PDF路径（使用一个小的测试文件）
TEST_PDF = r"C:\Users\Kevin\Desktop\excel\致同\扫描资料\扫描资料\中区建设\中区西片中小企业\其他文件（根据具体包含的纸质版文件命名）\中区西片法定图制.pdf"


def check_gpu_status():
    """检查GPU状态"""
    print("=" * 60)
    print("1. GPU状态检查")
    print("=" * 60)
    
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存总量: {total:.2f} GB")
        
        # 检查当前占用
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            used = parts[0].strip()
            total = parts[1].strip()
            util = parts[2].strip()
            print(f"显存占用: {used} MB / {total} MB")
            print(f"GPU利用率: {util}%")
            
            if int(used) > 3000:
                print("\n⚠️ 警告: 显存占用较高，建议关闭其他程序后再测试")
    else:
        print("❌ CUDA不可用!")
        return False
    
    return True


def test_ocr_speed(image_size=768, base_size=1024, dpi=150):
    """测试OCR速度"""
    print(f"\n{'=' * 60}")
    print(f"2. OCR速度测试 (image_size={image_size}, base_size={base_size}, dpi={dpi})")
    print("=" * 60)
    
    from pdf_processor import PDFProcessor
    from deepseek_ocr2_engine import DeepSeekOCR2Engine
    import torch
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        before_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"测试前显存: {before_mem:.0f} MB")
    
    # 初始化
    print("\n初始化组件...")
    init_start = time.time()
    
    pdf_processor = PDFProcessor({'dpi': dpi})
    ocr_engine = DeepSeekOCR2Engine({})
    
    # 修改OCR参数（如果需要）
    # 这里我们无法直接修改，但可以测试默认配置
    
    init_time = time.time() - init_start
    print(f"初始化耗时: {init_time:.2f}s")
    
    if torch.cuda.is_available():
        after_init_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"模型加载后显存: {after_init_mem:.0f} MB")
    
    # 读取测试PDF
    if not os.path.exists(TEST_PDF):
        print(f"❌ 测试文件不存在: {TEST_PDF}")
        return None
    
    print(f"\n处理测试文件: {Path(TEST_PDF).name}")
    
    # OCR测试
    page_times = []
    total_chars = 0
    
    for page_num, page_image in pdf_processor.process_pdf(TEST_PDF):
        print(f"  第{page_num}页...", end=" ", flush=True)
        
        page_start = time.time()
        result = ocr_engine.recognize_pdf_page(page_image, page_num)
        page_time = time.time() - page_start
        
        text = result.get_full_text()
        chars = len(text)
        total_chars += chars
        page_times.append(page_time)
        
        print(f"{page_time:.2f}s ({chars}字)")
    
    # 统计结果
    avg_time = sum(page_times) / len(page_times) if page_times else 0
    total_time = sum(page_times)
    
    print(f"\n{'─' * 40}")
    print(f"处理页数: {len(page_times)}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均每页: {avg_time:.2f}s")
    print(f"识别字符: {total_chars}")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值显存: {peak_mem:.0f} MB")
    
    return {
        'pages': len(page_times),
        'total_time': total_time,
        'avg_time': avg_time,
        'chars': total_chars
    }


def run_comparison_test():
    """运行对比测试"""
    print("\n" + "=" * 60)
    print("3. 配置对比测试")
    print("=" * 60)
    
    configs = [
        {'name': '默认配置', 'dpi': 200},
        {'name': '低DPI配置', 'dpi': 150},
        {'name': '极低DPI配置', 'dpi': 100},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n测试: {config['name']} (DPI={config['dpi']})")
        print("-" * 40)
        
        result = test_ocr_speed(dpi=config['dpi'])
        if result:
            result['config'] = config['name']
            results.append(result)
    
    # 打印对比表
    if results:
        print("\n" + "=" * 60)
        print("对比结果")
        print("=" * 60)
        print(f"{'配置':<15} {'页数':<6} {'总耗时':<10} {'平均/页':<10} {'字符数':<8}")
        print("-" * 60)
        for r in results:
            print(f"{r['config']:<15} {r['pages']:<6} {r['total_time']:.2f}s     {r['avg_time']:.2f}s     {r['chars']:<8}")


def main():
    print("\n" + "=" * 60)
    print("PDF OCR 性能测试")
    print("=" * 60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 检查GPU
    if not check_gpu_status():
        print("\n请检查CUDA环境后重试")
        return
    
    # 2. 单次测试
    input("\n按Enter开始OCR速度测试...")
    result = test_ocr_speed()
    
    # 3. 询问是否进行对比测试
    if result:
        response = input("\n是否进行不同DPI配置的对比测试? (y/n): ")
        if response.lower() == 'y':
            run_comparison_test()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    print("\n优化建议:")
    print("1. 关闭后台程序释放显存（如Wallpaper Engine、浏览器等）")
    print("2. 降低DPI到150可提速约20-30%")
    print("3. 如果显存不足，可考虑使用PaddleOCR（速度更快但精度略低）")


if __name__ == "__main__":
    main()
