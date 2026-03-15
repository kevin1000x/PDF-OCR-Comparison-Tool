"""
混合OCR引擎模块
结合RapidOCR速度与DeepSeek-OCR2精度
=============================================

支持三种模式:
- smart: 智能切换 (RapidOCR先跑，低置信度用DeepSeek复核)
- rapid_only: 仅使用RapidOCR (最快，ONNX加速)
- deepseek_only: 仅使用DeepSeek-OCR2 (最准)

RapidOCR基于ONNX Runtime，与PyTorch完全兼容，无DLL冲突问题。
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field
from PIL import Image

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
    engine_used: str = ""
    
    def get_full_text(self) -> str:
        if not self.full_text:
            self.full_text = "\n".join([r.text for r in self.results])
        return self.full_text


class HybridOCREngine:
    """
    混合OCR引擎 - 结合RapidOCR速度与DeepSeek-OCR2精度
    
    Args:
        mode: 运行模式
            - 'smart': 智能切换 (默认) - RapidOCR快速识别，低置信度用DeepSeek
            - 'rapid_only': 仅RapidOCR (最快)
            - 'deepseek_only': 仅DeepSeek-OCR2 (最准)
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
        self._rapid_engine = None
        self._deepseek_engine = None
        self._rapid_available = None
        
        # 统计信息
        self.stats = {
            'rapid_calls': 0,
            'deepseek_calls': 0,
            'fallback_count': 0,
            'total_time': 0.0
        }
        
        logger.info(f"HybridOCREngine initialized: mode={mode}, threshold={confidence_threshold}")
    
    def _check_rapid_available(self) -> bool:
        """检查RapidOCR是否可用"""
        if self._rapid_available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._rapid_available = True
                logger.info("RapidOCR is available")
            except ImportError as e:
                self._rapid_available = False
                logger.warning(f"RapidOCR not available: {e}")
            except Exception as e:
                self._rapid_available = False
                logger.warning(f"RapidOCR unavailable: {e}")
        return self._rapid_available
    
    @property
    def rapid_engine(self):
        """延迟加载RapidOCR引擎"""
        if self._rapid_engine is None:
            if not self._check_rapid_available():
                return None
            
            logger.info("Loading RapidOCR engine...")
            try:
                from rapidocr_engine import RapidOCREngine
                self._rapid_engine = RapidOCREngine(self.config)
                logger.info("RapidOCR loaded")
            except Exception as e:
                logger.warning(f"Failed to load RapidOCR: {e}")
                self._rapid_available = False
                return None
        return self._rapid_engine
    
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
        """识别图像中的文本"""
        start_time = time.time()
        
        if self.mode == 'rapid_only':
            results = self._rapid_recognize(image)
        elif self.mode == 'deepseek_only':
            results = self._deepseek_recognize(image)
        else:  # smart mode
            results = self._smart_recognize(image)
        
        self.stats['total_time'] += time.time() - start_time
        return results
    
    def recognize_pdf_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """识别PDF页面"""
        start_time = time.time()
        
        if self.mode == 'rapid_only':
            result = self._rapid_recognize_page(page_image, page_num)
        elif self.mode == 'deepseek_only':
            result = self._deepseek_recognize_page(page_image, page_num)
        else:  # smart mode
            result = self._smart_recognize_page(page_image, page_num)
        
        self.stats['total_time'] += time.time() - start_time
        return result
    
    def _smart_recognize(self, image) -> List[OCRResult]:
        """智能模式: RapidOCR先跑，低置信度用DeepSeek复核"""
        if not self._check_rapid_available():
            logger.info("RapidOCR unavailable, using DeepSeek directly")
            return self._deepseek_recognize(image)
        
        try:
            rapid_results = self._rapid_recognize(image)
        except Exception as e:
            logger.warning(f"RapidOCR failed: {e}, using DeepSeek")
            return self._deepseek_recognize(image)
        
        # 计算平均置信度
        if rapid_results:
            avg_conf = sum(r.confidence for r in rapid_results) / len(rapid_results)
        else:
            avg_conf = 0.0
        
        # 置信度低于阈值，使用DeepSeek复核
        if avg_conf < self.threshold:
            logger.info(f"RapidOCR conf {avg_conf:.2f} < {self.threshold}, using DeepSeek")
            self.stats['fallback_count'] += 1
            return self._deepseek_recognize(image)
        
        return rapid_results
    
    def _smart_recognize_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """智能模式识别页面"""
        if not self._check_rapid_available():
            logger.info(f"Page {page_num}: RapidOCR unavailable, using DeepSeek")
            return self._deepseek_recognize_page(page_image, page_num)
        
        try:
            rapid_result = self._rapid_recognize_page(page_image, page_num)
        except Exception as e:
            logger.warning(f"Page {page_num}: RapidOCR failed: {e}, using DeepSeek")
            return self._deepseek_recognize_page(page_image, page_num)
        
        if rapid_result.avg_confidence < self.threshold:
            logger.info(f"Page {page_num}: RapidOCR conf {rapid_result.avg_confidence:.2f} < {self.threshold}, using DeepSeek")
            self.stats['fallback_count'] += 1
            return self._deepseek_recognize_page(page_image, page_num)
        
        return rapid_result
    
    def _rapid_recognize(self, image) -> List[OCRResult]:
        """使用RapidOCR识别"""
        self.stats['rapid_calls'] += 1
        
        if isinstance(image, str):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        results = self.rapid_engine.recognize_image(image)
        return results
    
    def _rapid_recognize_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """使用RapidOCR识别页面"""
        self.stats['rapid_calls'] += 1
        result = self.rapid_engine.recognize_pdf_page(page_image, page_num)
        result.engine_used = 'rapidocr'
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
        result.avg_confidence = 1.0
        result.engine_used = 'deepseek'
        return result
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_calls = self.stats['rapid_calls'] + self.stats['deepseek_calls']
        return {
            **self.stats,
            'total_calls': total_calls,
            'rapid_ratio': self.stats['rapid_calls'] / total_calls if total_calls > 0 else 0,
            'avg_time_per_call': self.stats['total_time'] / total_calls if total_calls > 0 else 0
        }
    
    def unload_deepseek(self):
        """卸载DeepSeek模型以释放显存"""
        if self._deepseek_engine:
            self._deepseek_engine.unload_model()
            self._deepseek_engine = None
            logger.info("DeepSeek engine unloaded from Hybrid engine")

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("HybridOCREngine Statistics:")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Total calls: {stats['total_calls']}")
        logger.info(f"  RapidOCR calls: {stats['rapid_calls']} ({stats['rapid_ratio']:.1%})")
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
        mode: 运行模式 ('smart', 'rapid_only', 'deepseek_only')
        confidence_threshold: 置信度阈值
        config: 额外配置
        
    Returns:
        HybridOCREngine实例
    """
    return HybridOCREngine(mode, confidence_threshold, config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = get_hybrid_engine(mode='smart', confidence_threshold=0.85)
    
    from PIL import Image, ImageDraw
    test_image = Image.new('RGB', (400, 100), 'white')
    draw = ImageDraw.Draw(test_image)
    draw.text((10, 30), "测试文本 Test 123", fill='black')
    
    results = engine.recognize_image(test_image)
    print(f"Results: {len(results)} items")
    for r in results:
        print(f"  - {r.text} (conf: {r.confidence:.2f})")
    
    engine.print_stats()
