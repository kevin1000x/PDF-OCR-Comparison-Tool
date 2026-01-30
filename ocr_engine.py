"""
OCR识别引擎模块
使用PaddleOCR进行中文文档识别
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str  # 识别的文本
    confidence: float  # 置信度
    bbox: List[List[int]] = field(default_factory=list)  # 边界框坐标
    

@dataclass
class PageOCRResult:
    """单页OCR结果"""
    page_num: int  # 页码（从1开始）
    results: List[OCRResult] = field(default_factory=list)  # 识别结果列表
    full_text: str = ""  # 完整文本
    
    def get_full_text(self) -> str:
        """获取页面完整文本"""
        if not self.full_text:
            self.full_text = "\n".join([r.text for r in self.results])
        return self.full_text


class OCREngine:
    """OCR识别引擎"""
    
    def __init__(self, config: dict):
        """
        初始化OCR引擎
        
        Args:
            config: OCR配置字典
        """
        self.config = config
        self.use_gpu = config.get('use_gpu', True)
        self.language = config.get('language', 'ch')
        self.preprocessing = config.get('preprocessing', {})
        self._ocr = None
        
    def _init_paddleocr(self):
        """延迟初始化PaddleOCR"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 3.x API - 使用默认server模型（高精度）
                self._ocr = PaddleOCR(
                    use_doc_orientation_classify=False,  # 禁用文档方向分类（对扫描件不需要）
                    use_doc_unwarping=False,  # 禁用文档展平（对扫描件不需要）
                    use_textline_orientation=False,  # 禁用文本行方向检测
                )
                logger.info("PaddleOCR initialized (PP-OCRv5 server, high accuracy)")
            except ImportError:
                logger.error("PaddleOCR not installed. Please run: pip install paddleocr")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise
                
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            预处理后的图像
        """
        if not self.preprocessing:
            return image
            
        processed = image.copy()
        
        # 转为灰度图进行处理
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
            
        # 去噪
        if self.preprocessing.get('denoise', False):
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
        # 增强对比度
        if self.preprocessing.get('enhance_contrast', False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
        # 纠正倾斜
        if self.preprocessing.get('deskew', False):
            gray = self._deskew(gray)
            
        # 转回BGR格式
        if len(image.shape) == 3:
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            processed = gray
            
        return processed
        
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        纠正图像倾斜
        
        Args:
            image: 灰度图像
            
        Returns:
            纠正后的图像
        """
        # 使用霍夫变换检测直线
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            return image
            
        # 计算平均角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)
                
        if not angles:
            return image
            
        median_angle = np.median(angles)
        
        # 如果倾斜角度很小，不需要纠正
        if abs(median_angle) < 0.5:
            return image
            
        # 旋转图像
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
        
    def recognize_image(self, image: np.ndarray) -> List[OCRResult]:
        """
        识别单张图像
        
        Args:
            image: 输入图像（BGR格式或PIL Image）
            
        Returns:
            OCR结果列表
        """
        self._init_paddleocr()
        
        # 如果是PIL Image，转换为numpy数组
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # 预处理
        processed = self.preprocess_image(image)
        
        # OCR识别 - PaddleOCR 3.x API: predict(input=...)
        try:
            result = self._ocr.predict(input=processed)
            return self._parse_result_v3(result)
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return []
    
    def _parse_result_v3(self, result) -> List[OCRResult]:
        """解析PaddleOCR 3.x结果 - 官方API返回result对象列表"""
        ocr_results = []
        try:
            if result is None:
                return ocr_results
            
            # PaddleOCR 3.x predict()返回生成器或列表
            for res in result:
                # 尝试获取识别文本列表
                rec_texts = []
                rec_scores = []
                dt_polys = []
                
                # 方式1: 属性访问 (官方3.x格式)
                if hasattr(res, 'rec_text'):
                    rec_texts = self._to_list(getattr(res, 'rec_text', []))
                    rec_scores = self._to_list(getattr(res, 'rec_score', []))
                    dt_polys = self._to_list(getattr(res, 'dt_polys', []))
                # 方式2: 字典访问
                elif isinstance(res, dict):
                    rec_texts = self._to_list(res.get('rec_text', res.get('rec_texts', [])))
                    rec_scores = self._to_list(res.get('rec_score', res.get('rec_scores', [])))
                    dt_polys = self._to_list(res.get('dt_polys', res.get('boxes', [])))
                # 方式3: 下标访问
                elif hasattr(res, '__getitem__'):
                    try:
                        rec_texts = self._to_list(res.get('rec_text', []))
                        rec_scores = self._to_list(res.get('rec_score', []))
                        dt_polys = self._to_list(res.get('dt_polys', []))
                    except:
                        pass
                
                # 解析每个识别结果
                for i in range(len(rec_texts)):
                    text = rec_texts[i]
                    # 安全检查文本是否有效
                    text_str = str(text) if text is not None else ""
                    if text_str and text_str.strip():
                        conf = rec_scores[i] if i < len(rec_scores) else 0.0
                        bbox = dt_polys[i] if i < len(dt_polys) else []
                        # 安全转换置信度
                        try:
                            conf_float = float(conf) if conf is not None else 0.0
                        except:
                            conf_float = 0.0
                        ocr_results.append(OCRResult(
                            text=text_str,
                            confidence=conf_float,
                            bbox=self._to_list(bbox)
                        ))
                        
        except Exception as e:
            logger.warning(f"Failed to parse OCR result: {e}, result type: {type(result)}")
            # 打印第一个结果的结构用于调试
            try:
                if result:
                    for res in result:
                        logger.debug(f"Result item type: {type(res)}, dir: {dir(res)[:10]}")
                        break
            except:
                pass
        return ocr_results
    
    def _to_list(self, val):
        """安全地将值转换为列表"""
        if val is None:
            return []
        # 处理numpy数组
        if hasattr(val, 'tolist'):
            return val.tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val]
        
    def recognize_image_file(self, image_path: str) -> List[OCRResult]:
        """
        识别图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            OCR结果列表
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return []
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
            
        return self.recognize_image(image)
        
    def recognize_pdf_page(self, page_image: Image.Image, page_num: int) -> PageOCRResult:
        """
        识别PDF的单页
        
        Args:
            page_image: PDF页面图像（PIL Image格式）
            page_num: 页码（从1开始）
            
        Returns:
            页面OCR结果
        """
        results = self.recognize_image(page_image)
        page_result = PageOCRResult(
            page_num=page_num,
            results=results
        )
        page_result.full_text = page_result.get_full_text()
        return page_result
        
    def batch_recognize(self, images: List[np.ndarray]) -> List[List[OCRResult]]:
        """
        批量识别图像
        
        Args:
            images: 图像列表
            
        Returns:
            每张图像的OCR结果列表
        """
        results = []
        for image in images:
            results.append(self.recognize_image(image))
        return results


class OCRResultExtractor:
    """OCR结果特征提取器"""
    
    # 日期正则模式
    DATE_PATTERNS = [
        r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?',
        r'\d{4}\.\d{1,2}\.\d{1,2}',
        r'\d{4}年\d{1,2}月\d{1,2}日',
    ]
    
    # 金额正则模式
    AMOUNT_PATTERNS = [
        r'[¥￥]\s*[\d,]+\.?\d*',
        r'[\d,]+\.?\d*\s*[元万]',
        r'金额[：:]\s*[\d,]+\.?\d*',
        r'合计[：:]\s*[\d,]+\.?\d*',
    ]
    
    # 编号正则模式
    NUMBER_PATTERNS = [
        r'[A-Za-z]*\d{4,}',
        r'编号[：:]\s*\S+',
        r'合同号[：:]\s*\S+',
        r'发票号[：:]\s*\S+',
    ]
    
    def __init__(self):
        import re
        self.re = re
        
    def extract_dates(self, text: str) -> List[str]:
        """提取日期"""
        dates = []
        for pattern in self.DATE_PATTERNS:
            matches = self.re.findall(pattern, text)
            dates.extend(matches)
        return list(set(dates))
        
    def extract_amounts(self, text: str) -> List[str]:
        """提取金额"""
        amounts = []
        for pattern in self.AMOUNT_PATTERNS:
            matches = self.re.findall(pattern, text)
            amounts.extend(matches)
        return list(set(amounts))
        
    def extract_numbers(self, text: str) -> List[str]:
        """提取编号"""
        numbers = []
        for pattern in self.NUMBER_PATTERNS:
            matches = self.re.findall(pattern, text)
            numbers.extend(matches)
        return list(set(numbers))
        
    def extract_features(self, text: str) -> Dict[str, List[str]]:
        """
        提取文本中的关键特征
        
        Args:
            text: OCR识别的文本
            
        Returns:
            特征字典
        """
        return {
            'dates': self.extract_dates(text),
            'amounts': self.extract_amounts(text),
            'numbers': self.extract_numbers(text),
            'keywords': self._extract_keywords(text)
        }
        
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 文本
            top_n: 返回前N个关键词
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取：根据词频
        import jieba
        words = jieba.lcut(text)
        # 过滤停用词和短词
        filtered = [w for w in words if len(w) >= 2 and not w.isdigit()]
        # 统计词频
        from collections import Counter
        word_counts = Counter(filtered)
        # 返回高频词
        return [w for w, _ in word_counts.most_common(top_n)]


if __name__ == "__main__":
    # 测试代码
    config = {
        'use_gpu': True,
        'language': 'ch',
        'preprocessing': {
            'deskew': True,
            'denoise': True,
            'enhance_contrast': True
        }
    }
    
    engine = OCREngine(config)
    print("OCR Engine initialized successfully.")
