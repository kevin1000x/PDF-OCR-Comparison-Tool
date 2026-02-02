"""测试RapidOCR和混合引擎"""
from rapidocr_engine import RapidOCREngine
from hybrid_ocr_engine import get_hybrid_engine
from PIL import Image, ImageDraw
import time

# 创建测试图像
img = Image.new('RGB', (400, 100), 'white')
draw = ImageDraw.Draw(img)
draw.text((10, 30), '发票号码: 12345678', fill='black')

print("=" * 50)
print("1. RapidOCR 测试")
print("=" * 50)

engine = RapidOCREngine()
start = time.time()
results = engine.recognize_image(img)
elapsed = time.time() - start

print(f"  Results: {len(results)} items")
for r in results:
    print(f"  - \"{r.text}\" (conf: {r.confidence:.2f})")
print(f"  Time: {elapsed:.3f}s")

print()
print("=" * 50)
print("2. 混合引擎测试 (smart mode)")
print("=" * 50)

hybrid = get_hybrid_engine(mode='smart', confidence_threshold=0.85)
start = time.time()
results = hybrid.recognize_image(img)
elapsed = time.time() - start

print(f"  Results: {len(results)} items")
for r in results:
    print(f"  - \"{r.text}\" (conf: {r.confidence:.2f})")
print(f"  Time: {elapsed:.3f}s")

stats = hybrid.get_stats()
print(f"\n  Stats: RapidOCR={stats['rapid_calls']}, DeepSeek={stats['deepseek_calls']}, Fallback={stats['fallback_count']}")

print()
print("✅ 测试完成")
