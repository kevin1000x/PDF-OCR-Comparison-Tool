"""
GPU加速测试脚本 - 修复版
"""

import torch
import time
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def test_gpu_status():
    """检查GPU状态"""
    print("=" * 60)
    print("1️⃣  GPU状态检查")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   PyTorch已分配: {allocated:.2f} GB")
        return True
    else:
        print("❌ CUDA不可用！")
        return False

def test_gpu_speed():
    """测试GPU计算速度"""
    print("\n" + "=" * 60)
    print("2️⃣  GPU vs CPU 速度对比")
    print("=" * 60)
    
    size = 4000
    print(f"\n测试矩阵运算 ({size}x{size})...")
    
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start = time.time()
    for _ in range(3):
        c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = (time.time() - start) / 3
    print(f"   CPU: {cpu_time*1000:.1f} ms")
    
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(3):
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 3
        print(f"   GPU: {gpu_time*1000:.1f} ms")
        print(f"   🚀 GPU加速: {cpu_time/gpu_time:.1f}x")
        
        del a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()

def test_ocr_speed():
    """测试DeepSeek-OCR2速度"""
    print("\n" + "=" * 60)
    print("3️⃣  DeepSeek-OCR2 速度测试")
    print("=" * 60)
    
    # 创建测试图片
    img = Image.new('RGB', (1200, 800), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", 36)
    except:
        font = ImageFont.load_default()
    
    test_text = """深圳市南山区高新区中区西片区地区
更新单元规划
CN07-2024-041/01
技术文件

项目概况
本项目位于南山区高新区中区西片区，总占地面积约5.2公顷
规划建筑面积约15万平方米，其中住宅约10万平方米
商业配套约3万平方米，公共设施约2万平方米"""
    
    draw.text((50, 50), test_text, fill='black', font=font)
    test_path = 'test_ocr_image.png'
    img.save(test_path)
    print(f"   创建测试图片: {test_path}")
    
    try:
        from deepseek_ocr2_engine import DeepSeekOCR2Engine
        
        print("\n   加载OCR引擎...")
        start = time.time()
        ocr = DeepSeekOCR2Engine()
        load_time = time.time() - start
        print(f"   引擎初始化: {load_time:.2f}s")
        
        # 使用正确的方法名
        print("\n   开始OCR测试（3次取平均）...")
        times = []
        
        for i in range(3):
            start = time.time()
            # 使用 recognize_image 方法
            results = ocr.recognize_image(test_path)
            elapsed = time.time() - start
            times.append(elapsed)
            
            # 提取文本
            if results:
                text = "\n".join([r.text for r in results])
                char_count = len(text)
            else:
                char_count = 0
            
            print(f"   第{i+1}次: {elapsed:.2f}s (识别{char_count}字符)")
        
        avg_time = sum(times) / len(times)
        print(f"\n   ⏱️  平均OCR时间: {avg_time:.2f}s/页")
        
        # 显示GPU显存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   📊 GPU显存使用: {allocated:.2f} GB")
        
        return avg_time
        
    except Exception as e:
        import traceback
        print(f"   ❌ 测试失败: {e}")
        traceback.print_exc()
        return None
    finally:
        # 清理
        if os.path.exists(test_path):
            os.remove(test_path)

def estimate_remaining_time(avg_time):
    """估算剩余时间"""
    print("\n" + "=" * 60)
    print("4️⃣  时间估算")
    print("=" * 60)
    
    if avg_time:
        # 假设情况
        scenarios = [
            ("已处理 334/523 页", 523 - 334),
            ("完整 523 页", 523),
            ("1000 页文档", 1000),
        ]
        
        for name, pages in scenarios:
            total_seconds = pages * avg_time
            hours = total_seconds / 3600
            print(f"   {name}: {hours:.1f} 小时")

def show_current_process():
    """显示当前运行的Python进程"""
    print("\n" + "=" * 60)
    print("5️⃣  当前OCR进程")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    
    for line in result.stdout.strip().split('\n'):
        if 'python' in line.lower():
            print(f"   {line}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔍 GPU加速完整测试")
    print("=" * 60)
    
    gpu_ok = test_gpu_status()
    
    if gpu_ok:
        test_gpu_speed()
        avg_time = test_ocr_speed()
        if avg_time:
            estimate_remaining_time(avg_time)
        show_current_process()
        
        print("\n" + "=" * 60)
        print("💡 优化建议")
        print("=" * 60)
        print("""
如果速度仍然较慢，可以：

1. 关闭Wallpaper Engine释放显存
   任务栏右键 -> 退出

2. 降低DPI (修改config.yaml)
   pdf:
     dpi: 150

3. 确保高性能模式
   Windows设置 -> 电源 -> 高性能
""")
