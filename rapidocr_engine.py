"""
RapidOCR引擎 - 快速轻量级OCR
================================

基于ONNX Runtime，与PyTorch兼容，可作为混合引擎的快速通道
速度约为DeepSeek的5-10倍
"""

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
    engine_used: str = "rapidocr"
    
    def get_full_text(self) -> str:
        """获取页面完整文本"""
        if not self.full_text:
            self.full_text = "\n".join([r.text for r in self.results])
        return self.full_text


class RapidOCREngine:
    """
    RapidOCR引擎 - 基于ONNX的快速OCR
    
    特点:
    - 与PyTorch兼容（无DLL冲突）
    - 速度快（ONNX Runtime优化）
    - 准确度接近PaddleOCR
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._ocr = None
        self._initialized = False
        
    def _init_ocr(self):
        """延迟初始化RapidOCR"""
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
                self._initialized = True
                logger.info("RapidOCR initialized (ONNX Runtime)")
            except ImportError as e:
                logger.error(f"RapidOCR not installed: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize RapidOCR: {e}")
                raise
    
    def recognize_image(self, image: Union[np.ndarray, Image.Image, str]) -> List[OCRResult]:
        """
        识别图像中的文本
        
        Args:
            image: 输入图像
            
        Returns:
            OCR结果列表
        """
        self._init_ocr()
        
        # 转换图像格式
        if isinstance(image, str):
            img = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        # 确保是RGB格式
        if len(img.shape) == 2:  # 灰度图
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        
        # 执行OCR
        result, elapse = self._ocr(img)
        
        ocr_results = []
        if result:
            for item in result:
                # RapidOCR返回格式: [bbox, text, confidence]
                bbox, text, confidence = item
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                ))
        
        return ocr_results
    
    def recognize_pdf_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """
        识别PDF页面
        
        Args:
            page_image: PDF页面图像
            page_num: 页码
            
        Returns:
            页面OCR结果
        """
        results = self.recognize_image(page_image)
        
        # 计算平均置信度
        if results:
            avg_conf = sum(r.confidence for r in results) / len(results)
        else:
            avg_conf = 0.0
        
        return PageOCRResult(
            page_num=page_num,
            results=results,
            full_text="\n".join([r.text for r in results]),
            avg_confidence=avg_conf,
            engine_used="rapidocr"
        )


def get_rapid_engine(config: dict = None) -> RapidOCREngine:
    """工厂函数: 获取RapidOCR引擎"""
    return RapidOCREngine(config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = RapidOCREngine()
    
    # 创建测试图像
    from PIL import Image, ImageDraw
    test_image = Image.new('RGB', (400, 100), 'white')
    draw = ImageDraw.Draw(test_image)
    draw.text((10, 30), "测试文本 Test 123", fill='black')
    
    results = engine.recognize_image(test_image)
    print(f"Results: {len(results)} items")
    for r in results:
        print(f"  - {r.text} (conf: {r.confidence:.2f})")
