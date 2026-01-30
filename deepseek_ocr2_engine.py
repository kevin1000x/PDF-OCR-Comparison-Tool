"""
DeepSeek-OCR2 引擎模块
高精度快速OCR识别 - 完整修复版
"""

import os
import re
import sys
import tempfile
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str  # 识别的文本
    confidence: float = 1.0  # 置信度
    bbox: List[int] = field(default_factory=list)  # 边界框 [x1, y1, x2, y2]


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


class DeepSeekOCR2Engine:
    """DeepSeek-OCR2 识别引擎"""
    
    def __init__(self, config: dict = None):
        """
        初始化OCR引擎
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._model = None
        self._tokenizer = None
        self._temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
        self._initialized = False
        logger.info(f"DeepSeekOCR2Engine temp dir: {self._temp_dir}")
        
    def _init_model(self):
        """延迟初始化模型"""
        if self._initialized:
            return
            
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        logger.info("Initializing DeepSeek-OCR2 model...")
        
        model_name = 'deepseek-ai/DeepSeek-OCR-2'
        
        try:
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # 加载模型 (bfloat16节省显存)
            self._model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            self._initialized = True
            logger.info("DeepSeek-OCR2 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek-OCR2: {e}")
            raise
    
    def recognize_image(self, image) -> List[OCRResult]:
        """
        识别图像中的文本
        
        Args:
            image: 图像 (numpy array, PIL Image, or file path)
            
        Returns:
            OCR结果列表
        """
        import numpy as np
        from PIL import Image as PILImage
        import cv2
        
        self._init_model()
        
        # 清理临时目录中的旧文件
        self._cleanup_temp_files()
        
        # 保存图像到临时文件
        image_path = self._save_image(image)
        if not image_path:
            logger.error("Failed to save image for OCR")
            return []
        
        # 运行OCR
        ocr_text = self._run_inference(image_path)
        
        # 解析结果
        results = self._parse_result(ocr_text)
        
        logger.debug(f"OCR found {len(results)} text segments")
        return results
    
    def _save_image(self, image) -> Optional[str]:
        """保存图像到临时文件"""
        import numpy as np
        from PIL import Image as PILImage
        import cv2
        
        try:
            temp_path = os.path.join(self._temp_dir, "input_image.png")
            
            if isinstance(image, str):
                # 如果是路径，直接使用
                if os.path.exists(image):
                    return image
                else:
                    logger.error(f"Image file not found: {image}")
                    return None
                    
            elif isinstance(image, np.ndarray):
                # OpenCV格式 (BGR) 或 RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    cv2.imwrite(temp_path, image)
                else:
                    cv2.imwrite(temp_path, image)
                return temp_path
                
            elif isinstance(image, PILImage.Image):
                image.save(temp_path)
                return temp_path
                
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None
    
    def _run_inference(self, image_path: str) -> str:
        """运行OCR推理"""
        prompt = "<image>\nFree OCR."  # 纯文本模式
        captured_output = ""
        result_from_file = ""
        
        try:
            # 方法1: 捕获stdout
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = self._model.infer(
                    self._tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=self._temp_dir,
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=True
                )
            
            captured_output = stdout_capture.getvalue()
            
            # 方法2: 检查返回值
            if result is not None:
                if isinstance(result, str) and result.strip():
                    return result
                elif hasattr(result, '__str__'):
                    result_str = str(result)
                    if result_str and result_str != 'None':
                        return result_str
            
            # 方法3: 从保存的文件读取
            result_from_file = self._read_result_files()
            
            # 选择最佳结果
            if result_from_file and len(result_from_file) > len(captured_output):
                return result_from_file
            elif captured_output:
                return captured_output
            else:
                return result_from_file
                
        except Exception as e:
            logger.error(f"OCR inference failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _read_result_files(self) -> str:
        """从临时目录读取结果文件"""
        result_text = ""
        
        # 尝试多种可能的结果文件名
        possible_files = [
            "result.md",
            "result.txt", 
            "output.md",
            "output.txt",
            "*.md",
            "*.txt"
        ]
        
        for pattern in possible_files:
            search_pattern = os.path.join(self._temp_dir, pattern)
            files = glob.glob(search_pattern)
            
            for file_path in files:
                if os.path.basename(file_path) == "input_image.png":
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content and len(content) > len(result_text):
                            result_text = content
                            logger.debug(f"Read OCR result from: {file_path}")
                except Exception as e:
                    logger.debug(f"Failed to read {file_path}: {e}")
        
        return result_text
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for f in os.listdir(self._temp_dir):
                if f != "input_image.png":
                    file_path = os.path.join(self._temp_dir, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        except Exception as e:
            logger.debug(f"Cleanup failed: {e}")
    
    def _parse_result(self, result: str) -> List[OCRResult]:
        """解析DeepSeek-OCR2的输出"""
        ocr_results = []
        
        if not result or not result.strip():
            logger.warning("Empty OCR result")
            return ocr_results
        
        # 方法1: 解析带位置标记的格式
        # 格式: <|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>\n文本
        pattern = r'<\|ref\|>text<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>\s*\n?(.+?)(?=<\|ref\|>|$)'
        matches = re.findall(pattern, result, re.DOTALL)
        
        for match in matches:
            try:
                x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
                text = match[4].strip()
                if text:
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=1.0,
                        bbox=[x1, y1, x2, y2]
                    ))
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse match: {e}")
        
        # 方法2: 如果没有匹配到位置格式，提取纯文本
        if not ocr_results:
            ocr_results = self._extract_plain_text(result)
        
        return ocr_results
    
    def _extract_plain_text(self, result: str) -> List[OCRResult]:
        """提取纯文本（无位置信息）"""
        ocr_results = []
        
        # 清理各种标记和调试信息
        clean_text = result
        
        # 移除XML/特殊标记
        clean_text = re.sub(r'<\|[^|]+\|>', '', clean_text)
        clean_text = re.sub(r'\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]', '', clean_text)
        
        # 移除调试输出
        debug_patterns = [
            r'={3,}\s*\n',  # ===\n
            r'BASE:.*?PATCHES:.*?\n',
            r'torch\.Size\([^)]+\)',
            r'The attention .*?\n',
            r'Setting .*?\n',
            r'`.*?` is deprecated.*?\n',
            r'Creating model:.*?\n',
            r'Model files.*?\n',
            r'Checking connectivity.*?\n',
        ]
        
        for pattern in debug_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        # 过滤关键词
        filter_keywords = [
            'attention', 'token', 'cache', 'warning', 'setting',
            'deprecated', 'torch.size', 'creating model', 'model files',
            'connectivity', 'eos_token', 'pad_token', 'position_ids'
        ]
        
        # 按行处理
        lines = clean_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过调试行
            line_lower = line.lower()
            if any(kw in line_lower for kw in filter_keywords):
                continue
            
            # 跳过太短的行（可能是噪声）
            if len(line) < 2:
                continue
            
            ocr_results.append(OCRResult(text=line, confidence=1.0))
        
        return ocr_results
    
    def recognize_pdf_page(self, page_image, page_num: int) -> PageOCRResult:
        """
        识别PDF页面
        
        Args:
            page_image: 页面图像
            page_num: 页码
            
        Returns:
            页面OCR结果
        """
        results = self.recognize_image(page_image)
        
        page_result = PageOCRResult(
            page_num=page_num,
            results=results
        )
        
        # 确保full_text被设置
        if results:
            page_result.full_text = "\n".join([r.text for r in results])
        
        return page_result
    
    def __del__(self):
        """清理临时目录"""
        try:
            import shutil
            if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except:
            pass
