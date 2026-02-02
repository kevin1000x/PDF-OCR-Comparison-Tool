"""
快速功能验证测试 - 验证核心匹配逻辑
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("PDF-OCR 核心功能验证")
print("=" * 60)

# 1. 测试ContentMatcher
print("\n[1] 测试 ContentMatcher...")
try:
    from content_matcher import ContentMatcher, PageFeatures

    matcher = ContentMatcher({
        'similarity_threshold': 0.75,
        'exact_match_threshold': 0.95
    })

    # 创建测试页面
    ref_page = PageFeatures(
        file_path="参照资料.pdf",
        page_num=1,
        text="深圳市高新技术产业园区建设工程施工合同 甲方：高新区管委会 乙方：建设公司 金额：500万元",
        doc_type="合同"
    )

    # 构建索引
    matcher.build_reference_index([ref_page])

    # 测试匹配
    voucher_page = PageFeatures(
        file_path="凭证.pdf",
        page_num=1,
        text="高新区建设工程款 施工合同付款 金额500万",
        doc_type="凭证"
    )

    matches = matcher.find_matches(voucher_page)

    if matches:
        print(f"  [OK] 匹配成功: 相似度 {matches[0][1]:.2%}")
    else:
        print("  [WARN] 未找到匹配（可能阈值较高）")

except Exception as e:
    print(f"  [FAIL] ContentMatcher: {e}")

# 2. 测试VoucherGrouper
print("\n[2] 测试 VoucherGrouper...")
try:
    from voucher_grouper import VoucherGrouper, VoucherGroup
    from content_matcher import PageFeatures

    grouper = VoucherGrouper()

    # 模拟页面
    pages = [
        PageFeatures("test.pdf", 1, "记账凭证 凭证号001 借方金额", "凭证"),
        PageFeatures("test.pdf", 2, "工程施工合同书", "合同"),
        PageFeatures("test.pdf", 3, "增值税专用发票", "发票"),
        PageFeatures("test.pdf", 4, "记账凭证 凭证号002", "凭证"),
    ]

    groups = grouper.group_pages(pages)
    summary = grouper.get_summary(groups)

    print(f"  [OK] 分组成功: {summary['voucher_count']} 个凭证组, {summary['total_attachments']} 个附件")

except Exception as e:
    print(f"  [FAIL] VoucherGrouper: {e}")

# 3. 测试VoucherGroupMatcher
print("\n[3] 测试 VoucherGroupMatcher...")
try:
    from content_matcher import ContentMatcher, VoucherGroupMatcher, PageFeatures
    from voucher_grouper import VoucherGroup

    matcher = ContentMatcher({'similarity_threshold': 0.5, 'exact_match_threshold': 0.9})

    # 构建参照索引
    ref_pages = [
        PageFeatures("ref.pdf", 1, "高新区建设工程施工合同 工程款支付", "合同"),
    ]
    matcher.build_reference_index(ref_pages)

    # 创建凭证组
    voucher_page = PageFeatures("voucher.pdf", 1, "记账凭证 高新区工程款", "凭证")
    attachment = PageFeatures("voucher.pdf", 2, "施工合同 高新区建设", "合同")

    group = VoucherGroup(voucher_page=voucher_page, attachment_pages=[attachment])

    # 匹配
    group_matcher = VoucherGroupMatcher(matcher)
    result = group_matcher.match_group(group)

    print(f"  [OK] 匹配结果: {result.match_status} (相似度: {result.similarity:.2%})")

except Exception as e:
    print(f"  [FAIL] VoucherGroupMatcher: {e}")

# 4. 测试HybridOCREngine
print("\n[4] 测试 HybridOCREngine...")
try:
    from hybrid_ocr_engine import get_hybrid_engine

    engine = get_hybrid_engine(mode='smart', confidence_threshold=0.85)
    print(f"  [OK] 混合引擎创建成功")
    print(f"       模式: {engine.mode}, 阈值: {engine.threshold}")

    # 检查RapidOCR可用性
    if engine._check_rapid_available():
        print("       RapidOCR: 可用")
    else:
        print("       RapidOCR: 不可用 (将使用DeepSeek)")

except Exception as e:
    print(f"  [FAIL] HybridOCREngine: {e}")

# 5. 测试run_ocr模块
print("\n[5] 测试 run_ocr 模块...")
try:
    from run_ocr import run_ocr_pipeline_with_callback
    print("  [OK] run_ocr_pipeline_with_callback 可导入")
except Exception as e:
    print(f"  [FAIL] run_ocr: {e}")

print("\n" + "=" * 60)
print("功能验证完成")
print("=" * 60)
