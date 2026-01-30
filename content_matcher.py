"""
内容比对模块
实现页面级别的文档匹配和重复检测
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageFeatures:
    """页面特征"""
    file_path: str  # 文件路径
    page_num: int   # 页码
    text: str       # 全文文本
    doc_type: str   # 文档类型
    dates: List[str] = field(default_factory=list)  # 日期
    amounts: List[str] = field(default_factory=list)  # 金额
    numbers: List[str] = field(default_factory=list)  # 编号
    keywords: List[str] = field(default_factory=list)  # 关键词
    text_hash: str = ""  # 文本哈希（用于快速比对）
    
    def __post_init__(self):
        """计算文本哈希"""
        import hashlib
        # 规范化文本后计算哈希
        normalized = re.sub(r'\s+', '', self.text.lower())
        self.text_hash = hashlib.md5(normalized.encode()).hexdigest()


@dataclass
class MatchResult:
    """匹配结果"""
    source_file: str      # 源文件路径
    source_pages: str     # 源文件页码范围（如 "P1-P3"）
    doc_type: str         # 文档类型
    match_status: str     # 匹配状态: "完全匹配", "部分匹配", "未找到"
    target_file: str      # 匹配的目标文件
    target_pages: str     # 目标文件页码范围
    similarity: float     # 相似度
    matched_keywords: List[str] = field(default_factory=list)  # 匹配的关键词


class TextSimilarity:
    """文本相似度计算"""
    
    @staticmethod
    def cosine_similarity(text1: str, text2: str) -> float:
        """
        计算余弦相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度 (0-1)
        """
        import jieba
        from collections import Counter
        import math
        
        # 分词
        words1 = list(jieba.cut(text1))
        words2 = list(jieba.cut(text2))
        
        # 词频统计
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        # 获取所有词
        all_words = set(counter1.keys()) | set(counter2.keys())
        
        # 计算向量
        vec1 = [counter1.get(w, 0) for w in all_words]
        vec2 = [counter2.get(w, 0) for w in all_words]
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """
        计算Jaccard相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度 (0-1)
        """
        import jieba
        
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
        
    @staticmethod
    def levenshtein_similarity(text1: str, text2: str) -> float:
        """
        计算基于编辑距离的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度 (0-1)
        """
        # 限制长度以提高性能
        text1 = text1[:1000]
        text2 = text2[:1000]
        
        m, n = len(text1), len(text2)
        
        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0
            
        # 动态规划计算编辑距离
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
        distance = dp[m][n]
        max_len = max(m, n)
        
        return 1 - (distance / max_len)


class PageFeatureIndex:
    """页面特征索引（用于快速检索）"""
    
    def __init__(self):
        self.pages: List[PageFeatures] = []
        self.hash_index: Dict[str, List[int]] = defaultdict(list)  # 哈希 -> 页面索引列表
        self.date_index: Dict[str, List[int]] = defaultdict(list)  # 日期 -> 页面索引列表
        self.amount_index: Dict[str, List[int]] = defaultdict(list)  # 金额 -> 页面索引列表
        
    def add_page(self, page: PageFeatures):
        """添加页面到索引"""
        idx = len(self.pages)
        self.pages.append(page)
        
        # 更新哈希索引
        self.hash_index[page.text_hash].append(idx)
        
        # 更新日期索引
        for date in page.dates:
            self.date_index[date].append(idx)
            
        # 更新金额索引
        for amount in page.amounts:
            # 规范化金额格式
            normalized_amount = re.sub(r'[,，\s]', '', amount)
            self.amount_index[normalized_amount].append(idx)
            
    def find_by_hash(self, text_hash: str) -> List[PageFeatures]:
        """根据哈希查找页面"""
        indices = self.hash_index.get(text_hash, [])
        return [self.pages[i] for i in indices]
        
    def find_by_date(self, date: str) -> List[PageFeatures]:
        """根据日期查找页面"""
        indices = self.date_index.get(date, [])
        return [self.pages[i] for i in indices]
        
    def find_by_amount(self, amount: str) -> List[PageFeatures]:
        """根据金额查找页面"""
        normalized = re.sub(r'[,，\s]', '', amount)
        indices = self.amount_index.get(normalized, [])
        return [self.pages[i] for i in indices]
        
    def get_all_pages(self) -> List[PageFeatures]:
        """获取所有页面"""
        return self.pages
        
    def get_pages_by_file(self, file_path: str) -> List[PageFeatures]:
        """获取指定文件的所有页面"""
        return [p for p in self.pages if p.file_path == file_path]


class ContentMatcher:
    """内容比对器"""
    
    def __init__(self, config: dict):
        """
        初始化内容比对器
        
        Args:
            config: matching配置字典
        """
        self.config = config.get('matching', {})
        self.algorithm = self.config.get('algorithm', 'cosine')
        self.exact_threshold = self.config.get('exact_match_threshold', 0.95)
        self.partial_threshold = self.config.get('partial_match_threshold', 0.60)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.75)
        
        # 初始化相似度计算器
        self.similarity_calculator = TextSimilarity()
        
        # 参照资料索引
        self.reference_index = PageFeatureIndex()
        
    def build_reference_index(self, pages: List[PageFeatures]):
        """
        构建参照资料索引
        
        Args:
            pages: 参照资料的页面特征列表
        """
        logger.info(f"Building reference index with {len(pages)} pages")
        self.reference_index = PageFeatureIndex()
        for page in pages:
            self.reference_index.add_page(page)
        logger.info("Reference index built successfully")
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度
        """
        if self.algorithm == 'cosine':
            return self.similarity_calculator.cosine_similarity(text1, text2)
        elif self.algorithm == 'jaccard':
            return self.similarity_calculator.jaccard_similarity(text1, text2)
        elif self.algorithm == 'levenshtein':
            return self.similarity_calculator.levenshtein_similarity(text1, text2)
        else:
            return self.similarity_calculator.cosine_similarity(text1, text2)
            
    def find_matches(self, source_page: PageFeatures) -> List[Tuple[PageFeatures, float]]:
        """
        为源页面查找匹配的参照页面
        
        Args:
            source_page: 源页面特征
            
        Returns:
            匹配的页面和相似度列表
        """
        matches = []
        
        # 1. 首先检查哈希精确匹配
        hash_matches = self.reference_index.find_by_hash(source_page.text_hash)
        for ref_page in hash_matches:
            matches.append((ref_page, 1.0))
            
        if matches:
            return matches  # 有精确匹配，直接返回
            
        # 2. 根据日期和金额快速筛选候选
        candidates = set()
        
        for date in source_page.dates:
            for ref_page in self.reference_index.find_by_date(date):
                candidates.add(id(ref_page))
                
        for amount in source_page.amounts:
            for ref_page in self.reference_index.find_by_amount(amount):
                candidates.add(id(ref_page))
                
        # 3. 如果没有候选，检查所有页面（限制数量）
        if not candidates:
            all_pages = self.reference_index.get_all_pages()
            # 限制比对数量以提高性能
            candidates = {id(p) for p in all_pages[:100]}
            
        # 4. 计算与候选页面的相似度
        for ref_page in self.reference_index.get_all_pages():
            if id(ref_page) in candidates:
                similarity = self.calculate_similarity(source_page.text, ref_page.text)
                if similarity >= self.partial_threshold:
                    matches.append((ref_page, similarity))
                    
        # 按相似度降序排序
        matches.sort(key=lambda x: -x[1])
        
        return matches[:5]  # 返回前5个最佳匹配
        
    def match_page(self, source_page: PageFeatures) -> MatchResult:
        """
        匹配单个页面
        
        Args:
            source_page: 源页面特征
            
        Returns:
            匹配结果
        """
        matches = self.find_matches(source_page)
        
        if not matches:
            return MatchResult(
                source_file=source_page.file_path,
                source_pages=f"P{source_page.page_num}",
                doc_type=source_page.doc_type,
                match_status="未找到",
                target_file="",
                target_pages="",
                similarity=0.0
            )
            
        best_match, similarity = matches[0]
        
        # 判断匹配状态
        if similarity >= self.exact_threshold:
            status = "完全匹配"
        elif similarity >= self.similarity_threshold:
            status = "部分匹配"
        else:
            status = "低相似度"
            
        # 找出匹配的关键词
        source_keywords = set(source_page.keywords)
        target_keywords = set(best_match.keywords)
        matched_keywords = list(source_keywords & target_keywords)
        
        return MatchResult(
            source_file=source_page.file_path,
            source_pages=f"P{source_page.page_num}",
            doc_type=source_page.doc_type,
            match_status=status,
            target_file=best_match.file_path,
            target_pages=f"P{best_match.page_num}",
            similarity=similarity,
            matched_keywords=matched_keywords
        )
        
    def match_document(self, source_pages: List[PageFeatures]) -> List[MatchResult]:
        """
        匹配整个文档的所有页面
        
        Args:
            source_pages: 源文档的页面特征列表
            
        Returns:
            匹配结果列表
        """
        results = []
        for page in source_pages:
            result = self.match_page(page)
            results.append(result)
        return results
        
    def generate_match_summary(self, results: List[MatchResult]) -> Dict:
        """
        生成匹配结果汇总
        
        Args:
            results: 匹配结果列表
            
        Returns:
            汇总信息字典
        """
        total = len(results)
        exact_matches = sum(1 for r in results if r.match_status == "完全匹配")
        partial_matches = sum(1 for r in results if r.match_status == "部分匹配")
        not_found = sum(1 for r in results if r.match_status == "未找到")
        
        return {
            'total_pages': total,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'not_found': not_found,
            'match_rate': (exact_matches + partial_matches) / total if total > 0 else 0
        }


class DocumentMatcher:
    """文档级别的匹配器（合并连续页面的匹配结果）"""
    
    def __init__(self, content_matcher: ContentMatcher):
        self.content_matcher = content_matcher
        
    def merge_page_results(self, page_results: List[MatchResult]) -> List[MatchResult]:
        """
        合并连续页面的匹配结果
        
        如果连续多页都匹配到同一个文档的连续页面，则合并为一个结果
        
        Args:
            page_results: 页面级别的匹配结果
            
        Returns:
            合并后的结果
        """
        if not page_results:
            return []
            
        merged = []
        current_group = [page_results[0]]
        
        for i in range(1, len(page_results)):
            prev = page_results[i-1]
            curr = page_results[i]
            
            # 检查是否可以合并（同一目标文件，连续页码）
            can_merge = (
                prev.target_file == curr.target_file and
                prev.match_status == curr.match_status and
                self._is_consecutive_pages(prev.target_pages, curr.target_pages)
            )
            
            if can_merge:
                current_group.append(curr)
            else:
                # 输出当前组
                merged.append(self._merge_group(current_group))
                current_group = [curr]
                
        # 处理最后一组
        if current_group:
            merged.append(self._merge_group(current_group))
            
        return merged
        
    def _is_consecutive_pages(self, pages1: str, pages2: str) -> bool:
        """检查两个页码是否连续"""
        try:
            # 从 "P1" 或 "P1-P3" 格式提取页码
            match1 = re.search(r'P(\d+)', pages1)
            match2 = re.search(r'P(\d+)', pages2)
            
            if match1 and match2:
                end1 = int(match1.group(1))
                start2 = int(match2.group(1))
                return start2 == end1 + 1
        except:
            pass
        return False
        
    def _merge_group(self, group: List[MatchResult]) -> MatchResult:
        """合并一组匹配结果"""
        if len(group) == 1:
            return group[0]
            
        first = group[0]
        last = group[-1]
        
        # 合并页码范围
        source_start = re.search(r'P(\d+)', first.source_pages).group(1)
        source_end = re.search(r'P(\d+)', last.source_pages).group(1)
        
        target_pages = ""
        if first.target_file:
            target_start = re.search(r'P(\d+)', first.target_pages).group(1)
            target_end = re.search(r'P(\d+)', last.target_pages).group(1)
            target_pages = f"P{target_start}-P{target_end}" if target_start != target_end else f"P{target_start}"
            
        # 计算平均相似度
        avg_similarity = sum(r.similarity for r in group) / len(group)
        
        # 合并关键词
        all_keywords = []
        for r in group:
            all_keywords.extend(r.matched_keywords)
        unique_keywords = list(set(all_keywords))
        
        return MatchResult(
            source_file=first.source_file,
            source_pages=f"P{source_start}-P{source_end}" if source_start != source_end else f"P{source_start}",
            doc_type=first.doc_type,
            match_status=first.match_status,
            target_file=first.target_file,
            target_pages=target_pages,
            similarity=avg_similarity,
            matched_keywords=unique_keywords
        )


if __name__ == "__main__":
    # 测试代码
    config = {
        'matching': {
            'algorithm': 'cosine',
            'exact_match_threshold': 0.95,
            'partial_match_threshold': 0.60,
            'similarity_threshold': 0.75
        }
    }
    
    matcher = ContentMatcher(config)
    
    # 创建测试数据
    ref_page = PageFeatures(
        file_path="reference/test.pdf",
        page_num=1,
        text="生物孵化器SARS项目加固工程款 日期：2003-05-31 金额：300000元",
        doc_type="凭证",
        dates=["2003-05-31"],
        amounts=["300000元"],
        keywords=["生物孵化器", "SARS", "加固工程"]
    )
    
    source_page = PageFeatures(
        file_path="voucher/test.pdf",
        page_num=1,
        text="生物孵化器SARS项目加固工程款（首期）日期：2003-05-31 金额：300000元",
        doc_type="凭证",
        dates=["2003-05-31"],
        amounts=["300000元"],
        keywords=["生物孵化器", "SARS", "加固工程", "首期"]
    )
    
    # 构建索引
    matcher.build_reference_index([ref_page])
    
    # 测试匹配
    result = matcher.match_page(source_page)
    print(f"Match status: {result.match_status}")
    print(f"Target file: {result.target_file}:{result.target_pages}")
    print(f"Similarity: {result.similarity:.2f}")
    print(f"Matched keywords: {result.matched_keywords}")
