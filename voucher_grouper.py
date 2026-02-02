"""
凭证分组模块
=============

将扫描的PDF页面按凭证进行分组：
- 识别凭证首页（包含"记账凭证"、"收款凭证"、"付款凭证"等关键词）
- 将凭证首页之后的连续非凭证页面归为该凭证的附件

使用场景：
    凭证PDF扫描件通常包含：凭证首页 + 多张附件（发票、合同、审批单等）
    本模块将这些页面正确分组，便于后续匹配处理
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# 凭证首页识别关键词
VOUCHER_KEYWORDS = [
    "记账凭证",
    "收款凭证", 
    "付款凭证",
    "转账凭证",
    "记帐凭证",  # 繁体/异体
    "现金凭证",
    "银行凭证",
    "凭证号",
    "凭证字号",
    "借方金额",
    "贷方金额",
    "会计凭证",
]

# 凭证首页正则模式（更严格的匹配）
VOUCHER_PATTERNS = [
    r'记\s*账?\s*凭\s*证',
    r'收\s*款\s*凭\s*证',
    r'付\s*款\s*凭\s*证',
    r'转\s*账\s*凭\s*证',
    r'凭\s*证\s*[号字]\s*[：:]\s*\d+',
    r'借\s*方\s*金?\s*额',
    r'贷\s*方\s*金?\s*额',
]


@dataclass
class VoucherGroup:
    """
    凭证组 - 包含凭证首页及其附件
    
    Attributes:
        voucher_page: 凭证首页（PageFeatures对象）
        attachment_pages: 附件页面列表
        file_path: 源文件路径
        voucher_page_num: 凭证首页页码
        combined_text: 凭证组的合并文本（用于匹配）
    """
    voucher_page: 'PageFeatures'
    attachment_pages: List['PageFeatures'] = field(default_factory=list)
    file_path: str = ""
    voucher_page_num: int = 0
    _combined_text: str = field(default="", repr=False)
    
    def __post_init__(self):
        if self.voucher_page:
            self.file_path = self.voucher_page.file_path
            self.voucher_page_num = self.voucher_page.page_num
    
    @property
    def combined_text(self) -> str:
        """获取凭证组的合并文本"""
        if not self._combined_text:
            texts = [self.voucher_page.text]
            texts.extend([p.text for p in self.attachment_pages])
            self._combined_text = "\n\n".join(texts)
        return self._combined_text
    
    @property
    def total_pages(self) -> int:
        """总页数（凭证页 + 附件页）"""
        return 1 + len(self.attachment_pages)
    
    @property
    def page_range(self) -> str:
        """页码范围字符串，如 "P1-P4" """
        start = self.voucher_page_num
        end = start + len(self.attachment_pages)
        if end == start:
            return f"P{start}"
        return f"P{start}-P{end}"
    
    @property
    def attachment_range(self) -> str:
        """附件页码范围"""
        if not self.attachment_pages:
            return "-"
        start = self.attachment_pages[0].page_num
        end = self.attachment_pages[-1].page_num
        if start == end:
            return f"P{start}"
        return f"P{start}-P{end}"
    
    def get_all_dates(self) -> List[str]:
        """获取凭证组中的所有日期"""
        dates = list(self.voucher_page.dates)
        for p in self.attachment_pages:
            dates.extend(p.dates)
        return list(set(dates))
    
    def get_all_amounts(self) -> List[str]:
        """获取凭证组中的所有金额"""
        amounts = list(self.voucher_page.amounts)
        for p in self.attachment_pages:
            amounts.extend(p.amounts)
        return list(set(amounts))


class VoucherGrouper:
    """
    凭证分组器
    
    识别PDF页面中的凭证首页，并将后续非凭证页面归为附件。
    
    Args:
        keywords: 自定义凭证识别关键词列表（可选）
        min_keyword_match: 最少匹配关键词数量（默认1）
        use_patterns: 是否使用正则模式匹配（默认True）
    """
    
    def __init__(self, 
                 keywords: List[str] = None,
                 min_keyword_match: int = 1,
                 use_patterns: bool = True):
        self.keywords = keywords or VOUCHER_KEYWORDS
        self.min_keyword_match = min_keyword_match
        self.use_patterns = use_patterns
        
        # 编译正则模式
        self.patterns = [re.compile(p, re.IGNORECASE) for p in VOUCHER_PATTERNS]
        
        logger.info(f"VoucherGrouper initialized with {len(self.keywords)} keywords")
    
    def is_voucher_page(self, text: str) -> Tuple[bool, List[str]]:
        """
        判断页面是否为凭证首页
        
        Args:
            text: 页面OCR文本
            
        Returns:
            (是否为凭证页, 匹配到的关键词列表)
        """
        matched_keywords = []
        
        # 关键词匹配
        for keyword in self.keywords:
            if keyword in text:
                matched_keywords.append(keyword)
        
        # 正则模式匹配
        if self.use_patterns:
            for pattern in self.patterns:
                match = pattern.search(text)
                if match:
                    matched_keywords.append(match.group())
        
        # 去重
        matched_keywords = list(set(matched_keywords))
        
        is_voucher = len(matched_keywords) >= self.min_keyword_match
        
        return is_voucher, matched_keywords
    
    def group_pages(self, pages: List['PageFeatures']) -> List[VoucherGroup]:
        """
        将页面列表分组为凭证单元
        
        处理逻辑：
        1. 按文件路径分组（不同文件的页面不能混在一起）
        2. 对每个文件的页面按页码排序
        3. 识别凭证首页，将后续非凭证页面归为附件
        
        Args:
            pages: 所有页面的PageFeatures列表
            
        Returns:
            VoucherGroup列表
        """
        if not pages:
            return []
        
        # 按文件分组
        file_pages = {}
        for page in pages:
            file_path = page.file_path
            if file_path not in file_pages:
                file_pages[file_path] = []
            file_pages[file_path].append(page)
        
        # 对每个文件进行分组
        all_groups = []
        
        for file_path, file_page_list in file_pages.items():
            # 按页码排序
            sorted_pages = sorted(file_page_list, key=lambda p: p.page_num)
            
            # 分组处理
            groups = self._group_single_file(sorted_pages)
            all_groups.extend(groups)
            
            logger.info(f"File {file_path}: {len(sorted_pages)} pages -> {len(groups)} voucher groups")
        
        return all_groups
    
    def _group_single_file(self, pages: List['PageFeatures']) -> List[VoucherGroup]:
        """
        对单个文件的页面进行凭证分组
        
        Args:
            pages: 按页码排序的页面列表
            
        Returns:
            VoucherGroup列表
        """
        groups = []
        current_group: Optional[VoucherGroup] = None
        orphan_pages = []  # 没有归属凭证的页面
        
        for page in pages:
            is_voucher, matched = self.is_voucher_page(page.text)
            
            if is_voucher:
                # 发现新凭证首页
                logger.debug(f"Voucher page detected: P{page.page_num}, keywords: {matched}")
                
                # 保存之前的组
                if current_group:
                    groups.append(current_group)
                
                # 如果之前有孤立页面，创建一个特殊组
                if orphan_pages:
                    # 将孤立页面作为"无凭证"组的附件
                    logger.warning(f"Found {len(orphan_pages)} orphan pages before first voucher")
                    # 可以选择忽略或创建特殊组，这里选择忽略
                    orphan_pages = []
                
                # 创建新组
                current_group = VoucherGroup(voucher_page=page)
                
            else:
                # 非凭证页面
                if current_group:
                    # 作为当前凭证的附件
                    current_group.attachment_pages.append(page)
                    logger.debug(f"Attachment page: P{page.page_num} -> Voucher P{current_group.voucher_page_num}")
                else:
                    # 文件开头没有凭证页的情况
                    orphan_pages.append(page)
        
        # 保存最后一组
        if current_group:
            groups.append(current_group)
        
        # 处理完全没有凭证的文件（将所有页面作为一个组）
        if not groups and orphan_pages:
            logger.warning(f"No voucher found in file, treating all pages as one group")
            # 使用第一页作为"凭证页"
            first_page = orphan_pages[0]
            groups.append(VoucherGroup(
                voucher_page=first_page,
                attachment_pages=orphan_pages[1:]
            ))
        
        return groups
    
    def get_summary(self, groups: List[VoucherGroup]) -> dict:
        """
        获取分组统计摘要
        
        Args:
            groups: VoucherGroup列表
            
        Returns:
            统计信息字典
        """
        total_pages = sum(g.total_pages for g in groups)
        total_attachments = sum(len(g.attachment_pages) for g in groups)
        
        return {
            'voucher_count': len(groups),
            'total_pages': total_pages,
            'total_attachments': total_attachments,
            'avg_attachments': total_attachments / len(groups) if groups else 0,
            'files': list(set(g.file_path for g in groups))
        }


# 便捷函数
def group_voucher_pages(pages: List['PageFeatures']) -> List[VoucherGroup]:
    """快速分组函数"""
    grouper = VoucherGrouper()
    return grouper.group_pages(pages)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟PageFeatures
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class MockPageFeatures:
        file_path: str
        page_num: int
        text: str
        doc_type: str = ""
        dates: List[str] = field(default_factory=list)
        amounts: List[str] = field(default_factory=list)
        numbers: List[str] = field(default_factory=list)
        keywords: List[str] = field(default_factory=list)
    
    # 创建测试数据
    test_pages = [
        MockPageFeatures("test.pdf", 1, "记账凭证 凭证号：001 借方：银行存款 100000", dates=["2024-01-15"]),
        MockPageFeatures("test.pdf", 2, "工程施工合同 甲方：XX公司", dates=["2024-01-10"]),
        MockPageFeatures("test.pdf", 3, "增值税专用发票 金额：50000元", amounts=["50000元"]),
        MockPageFeatures("test.pdf", 4, "记账凭证 凭证号：002 贷方：应付账款 200000"),
        MockPageFeatures("test.pdf", 5, "付款申请单"),
    ]
    
    grouper = VoucherGrouper()
    groups = grouper.group_pages(test_pages)
    
    print(f"\n分组结果：共 {len(groups)} 个凭证组")
    for i, group in enumerate(groups):
        print(f"\n凭证组 {i+1}:")
        print(f"  凭证页: P{group.voucher_page_num}")
        print(f"  附件数: {len(group.attachment_pages)}")
        print(f"  页码范围: {group.page_range}")
        print(f"  附件范围: {group.attachment_range}")
    
    summary = grouper.get_summary(groups)
    print(f"\n统计: {summary}")
