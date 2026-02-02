"""综合测试脚本 - 验证所有模块导入和关键功能"""

import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('=' * 60)
print('PDF-OCR-Comparison-Tool 综合测试')
print('=' * 60)

errors = []
warnings = []

# 1. Test core imports
print('\n[1/8] 测试 content_matcher...')
try:
    from content_matcher import ContentMatcher, PageFeatures, VoucherGroupMatcher, VoucherGroupMatchResult
    print('  [OK] content_matcher 导入成功')
except Exception as e:
    errors.append(f'content_matcher: {e}')
    print(f'  [FAIL] content_matcher: {e}')

# 2. Test voucher_grouper
print('\n[2/8] 测试 voucher_grouper...')
try:
    from voucher_grouper import VoucherGrouper, VoucherGroup
    print('  [OK] voucher_grouper 导入成功')
except Exception as e:
    errors.append(f'voucher_grouper: {e}')
    print(f'  [FAIL] voucher_grouper: {e}')

# 3. Test hybrid_ocr_engine
print('\n[3/8] 测试 hybrid_ocr_engine...')
try:
    from hybrid_ocr_engine import HybridOCREngine, get_hybrid_engine
    print('  [OK] hybrid_ocr_engine 导入成功')
except Exception as e:
    errors.append(f'hybrid_ocr_engine: {e}')
    print(f'  [FAIL] hybrid_ocr_engine: {e}')

# 4. Test run_ocr
print('\n[4/8] 测试 run_ocr...')
try:
    from run_ocr import run_ocr_pipeline_with_callback
    print('  [OK] run_ocr 导入成功')
except Exception as e:
    errors.append(f'run_ocr: {e}')
    print(f'  [FAIL] run_ocr: {e}')

# 5. Test ocr_engine
print('\n[5/8] 测试 ocr_engine...')
try:
    from ocr_engine import OCRResultExtractor
    print('  [OK] OCRResultExtractor 导入成功')
except Exception as e:
    errors.append(f'ocr_engine: {e}')
    print(f'  [FAIL] ocr_engine: {e}')

# 6. Test rapidocr_engine
print('\n[6/8] 测试 rapidocr_engine...')
try:
    from rapidocr_engine import RapidOCREngine
    print('  [OK] rapidocr_engine 导入成功')
except Exception as e:
    errors.append(f'rapidocr_engine: {e}')
    print(f'  [FAIL] rapidocr_engine: {e}')

# 7. Test deepseek_ocr2_engine (不实际初始化模型)
print('\n[7/8] 测试 deepseek_ocr2_engine...')
try:
    from deepseek_ocr2_engine import DeepSeekOCR2Engine
    print('  [OK] deepseek_ocr2_engine 导入成功')
except Exception as e:
    errors.append(f'deepseek_ocr2_engine: {e}')
    print(f'  [FAIL] deepseek_ocr2_engine: {e}')

# 8. Test GUI imports (不创建窗口)
print('\n[8/8] 测试 ocr_gui_modern...')
try:
    # 只测试导入，不创建实际GUI
    import ocr_gui_modern
    print('  [OK] ocr_gui_modern 导入成功')
except Exception as e:
    errors.append(f'ocr_gui_modern: {e}')
    print(f'  [FAIL] ocr_gui_modern: {e}')

# 测试关键功能
print('\n' + '=' * 60)
print('功能验证测试')
print('=' * 60)

# 测试 ContentMatcher
print('\n测试 ContentMatcher 初始化...')
try:
    matcher = ContentMatcher({
        'similarity_threshold': 0.75,
        'exact_match_threshold': 0.95
    })
    print('  [OK] ContentMatcher 创建成功')
except Exception as e:
    errors.append(f'ContentMatcher init: {e}')
    print(f'  [FAIL] ContentMatcher: {e}')

# 测试 VoucherGrouper
print('\n测试 VoucherGrouper 初始化...')
try:
    grouper = VoucherGrouper()
    print('  [OK] VoucherGrouper 创建成功')
except Exception as e:
    errors.append(f'VoucherGrouper init: {e}')
    print(f'  [FAIL] VoucherGrouper: {e}')

# 测试 HybridOCREngine 创建（不加载模型）
print('\n测试 HybridOCREngine 初始化...')
try:
    engine = get_hybrid_engine(mode='smart', confidence_threshold=0.85)
    print('  [OK] HybridOCREngine 创建成功')
    print(f'       模式: {engine.mode}, 阈值: {engine.threshold}')
except Exception as e:
    errors.append(f'HybridOCREngine init: {e}')
    print(f'  [FAIL] HybridOCREngine: {e}')

# 测试 RapidOCR
print('\n测试 RapidOCR 可用性...')
try:
    from rapidocr_onnxruntime import RapidOCR
    print('  [OK] RapidOCR ONNX 可用')
except ImportError as e:
    warnings.append(f'RapidOCR not available: {e}')
    print(f'  [WARN] RapidOCR 未安装: {e}')

# 汇总
print('\n' + '=' * 60)
print('测试结果汇总')
print('=' * 60)

if errors:
    print(f'\n❌ 发现 {len(errors)} 个错误:')
    for err in errors:
        print(f'   - {err}')
else:
    print('\n✅ 所有模块导入成功!')

if warnings:
    print(f'\n⚠️  {len(warnings)} 个警告:')
    for warn in warnings:
        print(f'   - {warn}')

print('\n测试完成.')
sys.exit(1 if errors else 0)
