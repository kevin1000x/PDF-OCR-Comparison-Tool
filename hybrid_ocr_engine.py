"""
混合OCR引擎模块
结合PaddleOCR速度与DeepSeek-OCR2精度
=============================================

支持三种模式:
- smart: 智能切换 (Paddle先跑，低置信度用DeepSeek复核)
- paddle_only: 仅使用PaddleOCR (最快)
- deepseek_only: 仅使用DeepSeek-OCR2 (最准)
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float = 1.0
    bbox: List[int] = field(default_factory=list)


@dataclass
class PageOCRResult:
    """单页OCR结果"""
    page_num: int
    results: List[OCRResult] = field(default_factory=list)
    full_text: str = ""
    avg_confidence: float = 1.0
    engine_used: str = ""  # 使用的引擎
    
    def get_full_text(self) -> str:
        """获取页面完整文本"""
        if not self.full_text:
            self.full_text = "\n".join([r.text for r in self.results])
        return self.full_text


class HybridOCREngine:
    """
    混合OCR引擎 - 结合PaddleOCR速度与DeepSeek-OCR2精度
    
    Args:
        mode: 运行模式
            - 'smart': 智能切换 (默认)
            - 'paddle_only': 仅PaddleOCR
            - 'deepseek_only': 仅DeepSeek-OCR2
        confidence_threshold: 置信度阈值 (smart模式下使用)
        config: 额外配置
    """
    
    def __init__(self, 
                 mode: str = 'smart',
                 confidence_threshold: float = 0.85,
                 config: dict = None):
        self.mode = mode
        self.threshold = confidence_threshold
        self.config = config or {}
        
        # 延迟加载的引擎
        self._paddle_engine = None
        self._deepseek_engine = None
        
        # 统计信息
        self.stats = {
            'paddle_calls': 0,
            'deepseek_calls': 0,
            'fallback_count': 0,  # Paddle低置信度后调用DeepSeek的次数
            'total_time': 0.0
        }
        
        logger.info(f"HybridOCREngine initialized: mode={mode}, threshold={confidence_threshold}")
    
    @property
    def paddle_engine(self):
        """延迟加载PaddleOCR引擎"""
        if self._paddle_engine is None:
            logger.info("Loading PaddleOCR engine...")
            from ocr_engine import OCREngine
            self._paddle_engine = OCREngine({
                'use_gpu': True,
                'language': 'ch'
            })
            logger.info("PaddleOCR loaded")
        return self._paddle_engine
    
    @property
    def deepseek_engine(self):
        """延迟加载DeepSeek-OCR2引擎"""
        if self._deepseek_engine is None:
            logger.info("Loading DeepSeek-OCR2 engine...")
            from deepseek_ocr2_engine import DeepSeekOCR2Engine
            self._deepseek_engine = DeepSeekOCR2Engine(self.config)
            logger.info("DeepSeek-OCR2 loaded")
        return self._deepseek_engine
    
    def recognize_image(self, image: Union[np.ndarray, Image.Image, str]) -> List[OCRResult]:
        """
        识别图像中的文本
        
        Args:
            image: 输入图像 (numpy数组/PIL Image/文件路径)
            
        Returns:
            OCR结果列表
        """
        start_time = time.time()
        
        if self.mode == 'paddle_only':
            results = self._paddle_recognize(image)
        elif self.mode == 'deepseek_only':
            results = self._deepseek_recognize(image)
        else:  # smart mode
            results = self._smart_recognize(image)
        
        self.stats['total_time'] += time.time() - start_time
        return results
    
    def recognize_pdf_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """
        识别PDF页面
        
        Args:
            page_image: PDF页面图像
            page_num: 页码
            
        Returns:
            页面OCR结果
        """
        start_time = time.time()
        
        if self.mode == 'paddle_only':
            result = self._paddle_recognize_page(page_image, page_num)
        elif self.mode == 'deepseek_only':
            result = self._deepseek_recognize_page(page_image, page_num)
        else:  # smart mode
            result = self._smart_recognize_page(page_image, page_num)
        
        self.stats['total_time'] += time.time() - start_time
        return result
    
    def _smart_recognize(self, image) -> List[OCRResult]:
        """智能模式: Paddle先跑，低置信度用DeepSeek复核"""
        # 先用Paddle快速识别
        paddle_results = self._paddle_recognize(image)
        
        # 计算平均置信度
        if paddle_results:
            avg_conf = sum(r.confidence for r in paddle_results) / len(paddle_results)
        else:
            avg_conf = 0.0
        
        # 置信度低于阈值，使用DeepSeek复核
        if avg_conf < self.threshold:
            logger.info(f"Paddle confidence {avg_conf:.2f} < {self.threshold}, using DeepSeek")
            self.stats['fallback_count'] += 1
            return self._deepseek_recognize(image)
        
        return paddle_results
    
    def _smart_recognize_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """智能模式识别页面"""
        # 先用Paddle
        paddle_result = self._paddle_recognize_page(page_image, page_num)
        
        # 置信度低于阈值，使用DeepSeek
        if paddle_result.avg_confidence < self.threshold:
            logger.info(f"Page {page_num}: Paddle conf {paddle_result.avg_confidence:.2f} < {self.threshold}, using DeepSeek")
            self.stats['fallback_count'] += 1
            return self._deepseek_recognize_page(page_image, page_num)
        
        return paddle_result
    
    def _paddle_recognize(self, image) -> List[OCRResult]:
        """使用PaddleOCR识别"""
        self.stats['paddle_calls'] += 1
        
        # 转换图像格式
        if isinstance(image, str):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        results = self.paddle_engine.recognize_image(image)
        return results
    
    def _paddle_recognize_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """使用PaddleOCR识别页面"""
        self.stats['paddle_calls'] += 1
        result = self.paddle_engine.recognize_pdf_page(page_image, page_num)
        
        # 计算平均置信度
        if result.results:
            result.avg_confidence = sum(r.confidence for r in result.results) / len(result.results)
        else:
            result.avg_confidence = 0.0
        
        result.engine_used = 'paddle'
        return result
    
    def _deepseek_recognize(self, image) -> List[OCRResult]:
        """使用DeepSeek-OCR2识别"""
        self.stats['deepseek_calls'] += 1
        results = self.deepseek_engine.recognize_image(image)
        return results
    
    def _deepseek_recognize_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """使用DeepSeek-OCR2识别页面"""
        self.stats['deepseek_calls'] += 1
        result = self.deepseek_engine.recognize_pdf_page(page_image, page_num)
        result.avg_confidence = 1.0  # DeepSeek默认高置信度
        result.engine_used = 'deepseek'
        return result
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_calls = self.stats['paddle_calls'] + self.stats['deepseek_calls']
        return {
            **self.stats,
            'total_calls': total_calls,
            'paddle_ratio': self.stats['paddle_calls'] / total_calls if total_calls > 0 else 0,
            'avg_time_per_call': self.stats['total_time'] / total_calls if total_calls > 0 else 0
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("HybridOCREngine Statistics:")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Total calls: {stats['total_calls']}")
        logger.info(f"  Paddle calls: {stats['paddle_calls']} ({stats['paddle_ratio']:.1%})")
        logger.info(f"  DeepSeek calls: {stats['deepseek_calls']}")
        logger.info(f"  Fallback count: {stats['fallback_count']}")
        logger.info(f"  Total time: {stats['total_time']:.1f}s")
        logger.info(f"  Avg time/call: {stats['avg_time_per_call']:.2f}s")
        logger.info("=" * 50)


def get_hybrid_engine(mode: str = 'smart', 
                      confidence_threshold: float = 0.85,
                      config: dict = None) -> HybridOCREngine:
    """
    工厂函数: 获取混合OCR引擎
    
    Args:
        mode: 运行模式 ('smart', 'paddle_only', 'deepseek_only')
        confidence_threshold: 置信度阈值
        config: 额外配置
        
    Returns:
        HybridOCREngine实例
    """
    return HybridOCREngine(mode, confidence_threshold, config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试混合引擎
    engine = get_hybrid_engine(mode='smart', confidence_threshold=0.85)
    
    # 创建测试图像
    from PIL import Image, ImageDraw, ImageFont
    test_image = Image.new('RGB', (400, 100), 'white')
    draw = ImageDraw.Draw(test_image)
    draw.text((10, 30), "测试文本 Test 123", fill='black')
    
    # 测试识别
    results = engine.recognize_image(test_image)
    print(f"Results: {len(results)} items")
    for r in results:
        print(f"  - {r.text} (conf: {r.confidence:.2f})")
    
    engine.print_stats()
