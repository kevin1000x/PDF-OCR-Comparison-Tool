"""
文档分类器模块
使用规则匹配和本地大模型进行文档类型识别
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """文档类型枚举"""
    VOUCHER = "凭证"
    LETTER = "函"
    CONTRACT = "合同书"
    MEETING_MINUTES = "会议纪要"
    INVOICE = "发票"
    APPROVAL = "审批单"
    ENGINEERING_REPORT = "工程报告"
    BIDDING = "招标文件"
    FUND_APPLICATION = "资金申请"
    OTHER = "其他"


@dataclass
class ClassificationResult:
    """分类结果"""
    doc_type: str  # 文档类型
    confidence: float  # 置信度 (0-1)
    matched_keywords: List[str] = field(default_factory=list)  # 匹配的关键词
    reasoning: str = ""  # 分类理由（大模型返回）


class RuleBasedClassifier:
    """基于规则的文档分类器"""
    
    def __init__(self, config: dict):
        """
        初始化规则分类器
        
        Args:
            config: document_types配置字典
        """
        self.doc_types = config.get('document_types', {})
        self._build_patterns()
        
    def _build_patterns(self):
        """构建正则匹配模式"""
        self.patterns = {}
        for doc_type, type_config in self.doc_types.items():
            keywords = type_config.get('keywords', [])
            if keywords:
                # 构建正则模式，支持关键词匹配
                pattern = '|'.join(re.escape(k) for k in keywords)
                self.patterns[doc_type] = re.compile(pattern, re.IGNORECASE)
                
    def classify(self, text: str) -> ClassificationResult:
        """
        对文本进行分类
        
        Args:
            text: OCR识别的文本
            
        Returns:
            分类结果
        """
        scores = {}
        matched_keywords = {}
        
        for doc_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # 计算得分：匹配次数 * 关键词权重
                priority = self.doc_types.get(doc_type, {}).get('priority', 99)
                score = len(matches) * (100 - priority)
                scores[doc_type] = score
                matched_keywords[doc_type] = list(set(matches))
                
        if not scores:
            return ClassificationResult(
                doc_type="其他",
                confidence=0.5,
                matched_keywords=[]
            )
            
        # 选择得分最高的类型
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        # 计算置信度
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        
        return ClassificationResult(
            doc_type=best_type,
            confidence=min(confidence, 1.0),
            matched_keywords=matched_keywords.get(best_type, [])
        )


class LLMClassifier:
    """基于本地大模型的文档分类器"""
    
    def __init__(self, config: dict):
        """
        初始化大模型分类器
        
        Args:
            config: classification配置字典
        """
        self.model_name = config.get('model', 'Qwen/Qwen2.5-VL-7B-Instruct')
        self.device = config.get('device', 'cuda')
        self.max_new_tokens = config.get('max_new_tokens', 256)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self._model = None
        self._processor = None
        
        # 文档类型列表
        self.doc_types = [
            "凭证", "函", "合同书", "会议纪要", "发票",
            "审批单", "工程报告", "招标文件", "资金申请", "其他"
        ]
        
    def _init_model(self):
        """延迟初始化模型"""
        if self._model is not None:
            return
            
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self._model = self._model.cuda()
                
            self._model.eval()
            logger.info("Model loaded successfully")
            
        except ImportError:
            logger.error("transformers not installed. Please run: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def _build_prompt(self, text: str) -> str:
        """构建分类提示词"""
        doc_types_str = "、".join(self.doc_types)
        
        prompt = f"""你是一个专业的文档分类助手。请根据以下文档内容，判断其属于哪种类型。

可选的文档类型有：{doc_types_str}

文档内容：
{text[:2000]}  # 限制长度避免超出上下文

请按以下格式回答：
类型：[文档类型]
置信度：[0-100的数字]
理由：[简短说明]"""
        
        return prompt
        
    def classify(self, text: str, image=None) -> ClassificationResult:
        """
        使用大模型对文本/图像进行分类
        
        Args:
            text: OCR识别的文本
            image: 可选的页面图像（用于视觉理解）
            
        Returns:
            分类结果
        """
        self._init_model()
        
        prompt = self._build_prompt(text)
        
        try:
            import torch
            
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 解析响应
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return ClassificationResult(
                doc_type="其他",
                confidence=0.0,
                reasoning=f"分类失败: {e}"
            )
            
    def _parse_response(self, response: str) -> ClassificationResult:
        """解析模型响应"""
        doc_type = "其他"
        confidence = 0.5
        reasoning = ""
        
        # 提取类型
        type_match = re.search(r'类型[：:]\s*(.+?)(?:\n|$)', response)
        if type_match:
            extracted_type = type_match.group(1).strip()
            # 匹配到已知类型
            for dt in self.doc_types:
                if dt in extracted_type:
                    doc_type = dt
                    break
                    
        # 提取置信度
        conf_match = re.search(r'置信度[：:]\s*(\d+)', response)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100
            
        # 提取理由
        reason_match = re.search(r'理由[：:]\s*(.+?)(?:\n|$)', response)
        if reason_match:
            reasoning = reason_match.group(1).strip()
            
        return ClassificationResult(
            doc_type=doc_type,
            confidence=confidence,
            reasoning=reasoning
        )


class DocumentClassifier:
    """统一的文档分类器（组合规则和大模型）"""
    
    def __init__(self, config: dict):
        """
        初始化文档分类器
        
        Args:
            config: 完整配置字典
        """
        self.use_llm = config.get('classification', {}).get('use_llm', False)
        self.confidence_threshold = config.get('classification', {}).get('confidence_threshold', 0.7)
        
        # 初始化规则分类器
        self.rule_classifier = RuleBasedClassifier(config)
        
        # 按需初始化大模型分类器
        self._llm_classifier = None
        if self.use_llm:
            self._llm_config = config.get('classification', {})
            
    @property
    def llm_classifier(self) -> LLMClassifier:
        """延迟初始化大模型分类器"""
        if self._llm_classifier is None and self.use_llm:
            self._llm_classifier = LLMClassifier(self._llm_config)
        return self._llm_classifier
        
    def classify(self, text: str, image=None) -> ClassificationResult:
        """
        对文档进行分类
        
        策略：
        1. 先用规则分类器快速分类
        2. 如果置信度低于阈值且启用了大模型，则使用大模型
        
        Args:
            text: OCR文本
            image: 可选的页面图像
            
        Returns:
            分类结果
        """
        # 规则分类
        rule_result = self.rule_classifier.classify(text)
        
        # 如果置信度足够高，直接返回
        if rule_result.confidence >= self.confidence_threshold:
            logger.debug(f"Rule classification: {rule_result.doc_type} ({rule_result.confidence:.2f})")
            return rule_result
            
        # 如果启用大模型且规则分类置信度不够
        if self.use_llm and self.llm_classifier:
            try:
                llm_result = self.llm_classifier.classify(text, image)
                logger.debug(f"LLM classification: {llm_result.doc_type} ({llm_result.confidence:.2f})")
                
                # 如果大模型置信度更高，使用大模型结果
                if llm_result.confidence > rule_result.confidence:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to rule-based: {e}")
                
        return rule_result
        
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        批量分类
        
        Args:
            texts: 文本列表
            
        Returns:
            分类结果列表
        """
        return [self.classify(text) for text in texts]


# 文档类型的中文描述
DOC_TYPE_DESCRIPTIONS = {
    "凭证": "记账凭证、收款凭证、付款凭证等财务凭证",
    "函": "往来函件、复函、商洽函等公文",
    "合同书": "各类合同、协议、补充协议等",
    "会议纪要": "会议记录、会议决议等",
    "发票": "增值税发票、普通发票等",
    "审批单": "审批表、批复文件等",
    "工程报告": "验收报告、设计报告、施工报告等",
    "招标文件": "招标书、投标书、评标报告等",
    "资金申请": "资金申请表、用款申请、拨款单等",
    "其他": "无法归类的其他文档"
}


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    
    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 测试规则分类
        classifier = DocumentClassifier(config)
        
        test_texts = [
            "记账凭证 借方：银行存款 贷方：应收账款 金额：50000元",
            "关于高新区信息网扩容工程的函 致：深圳市规划局",
            "工程施工合同 甲方：XX公司 乙方：YY公司 合同金额：100万元",
            "会议纪要 时间：2024年1月15日 参会人员：张三、李四",
            "资金使用申请表 项目名称：SARS厂房装修 申请金额：30万元"
        ]
        
        for text in test_texts:
            result = classifier.classify(text)
            print(f"Text: {text[:30]}...")
            print(f"  Type: {result.doc_type}, Confidence: {result.confidence:.2f}")
            print(f"  Keywords: {result.matched_keywords}")
            print()
    else:
        print("Config file not found.")
