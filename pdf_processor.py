"""
PDF处理模块
提供PDF转图像、拆分、合并等功能
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PDFPageInfo:
    """PDF页面信息"""
    page_num: int  # 页码（从1开始）
    width: int     # 页面宽度
    height: int    # 页面高度


@dataclass 
class PDFInfo:
    """PDF文件信息"""
    file_path: str      # 文件路径
    page_count: int     # 总页数
    pages: List[PDFPageInfo]  # 页面信息列表


class PDFToImage:
    """PDF转图像工具"""
    
    def __init__(self, dpi: int = 200):
        """
        初始化PDF转图像工具
        
        Args:
            dpi: 图像分辨率
        """
        self.dpi = dpi
        self._poppler_path = None
        self._check_poppler()
        
    def _check_poppler(self):
        """检查Poppler是否可用"""
        import platform
        import glob
        
        if platform.system() == 'Windows':
            # 检查常见的Poppler安装路径
            possible_paths = [
                r"C:\poppler\Library\bin",
                r"C:\poppler\bin",
                r"C:\Program Files\poppler\bin",
                r"C:\Program Files\poppler\Library\bin",
            ]
            
            # 检查C:\poppler下的子目录（如 poppler-25.12.0\Library\bin）
            poppler_subdirs = glob.glob(r"C:\poppler\poppler-*\Library\bin")
            poppler_subdirs += glob.glob(r"C:\poppler\poppler-*\bin")
            possible_paths.extend(poppler_subdirs)
            
            for path in possible_paths:
                if os.path.exists(path):
                    self._poppler_path = path
                    logger.info(f"Found Poppler at: {path}")
                    break
                    
            if self._poppler_path is None:
                # 检查是否在PATH中
                import shutil
                if shutil.which('pdftoppm'):
                    logger.info("Using pdftoppm from system PATH")
                    # poppler_path=None 让pdf2image使用PATH中的
                else:
                    logger.warning(
                        "Poppler not found. Please install Poppler and add to PATH or "
                        "install to C:\\poppler. Download from: "
                        "https://github.com/oschwartz10612/poppler-windows/releases"
                    )
                
    def convert_page(self, pdf_path: str, page_num: int) -> Optional['Image.Image']:
        """
        将PDF的指定页转换为图像
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从1开始）
            
        Returns:
            PIL Image对象
        """
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(
                pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=self.dpi,
                poppler_path=self._poppler_path
            )
            
            if images:
                return images[0]
            return None
            
        except ImportError:
            logger.error("pdf2image not installed. Please run: pip install pdf2image")
            raise
        except Exception as e:
            logger.error(f"Failed to convert PDF page: {e}")
            return None
            
    def convert_all_pages(self, pdf_path: str) -> Generator['Image.Image', None, None]:
        """
        将PDF所有页转换为图像（生成器）
        
        Args:
            pdf_path: PDF文件路径
            
        Yields:
            PIL Image对象
        """
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                poppler_path=self._poppler_path
            )
            
            for image in images:
                yield image
                
        except ImportError:
            logger.error("pdf2image not installed. Please run: pip install pdf2image")
            raise
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            raise
            
    def convert_page_range(self, pdf_path: str, start: int, end: int) -> List['Image.Image']:
        """
        将PDF的指定页码范围转换为图像
        
        Args:
            pdf_path: PDF文件路径
            start: 起始页码（从1开始）
            end: 结束页码（包含）
            
        Returns:
            PIL Image对象列表
        """
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(
                pdf_path,
                first_page=start,
                last_page=end,
                dpi=self.dpi,
                poppler_path=self._poppler_path
            )
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF pages: {e}")
            return []


class PDFReader:
    """PDF读取器"""
    
    def __init__(self):
        self._fitz = None
        
    def _init_fitz(self):
        """延迟初始化PyMuPDF"""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                logger.error("PyMuPDF not installed. Please run: pip install PyMuPDF")
                raise
                
    def get_info(self, pdf_path: str) -> PDFInfo:
        """
        获取PDF信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            PDF信息对象
        """
        self._init_fitz()
        
        doc = self._fitz.open(pdf_path)
        pages = []
        
        for i in range(len(doc)):
            page = doc[i]
            rect = page.rect
            pages.append(PDFPageInfo(
                page_num=i + 1,
                width=int(rect.width),
                height=int(rect.height)
            ))
            
        info = PDFInfo(
            file_path=pdf_path,
            page_count=len(doc),
            pages=pages
        )
        
        doc.close()
        return info
        
    def get_page_count(self, pdf_path: str) -> int:
        """获取PDF页数"""
        self._init_fitz()
        doc = self._fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
        
    def extract_text(self, pdf_path: str, page_num: int) -> str:
        """
        提取PDF指定页的文本（如果有嵌入文本）
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从1开始）
            
        Returns:
            提取的文本
        """
        self._init_fitz()
        
        doc = self._fitz.open(pdf_path)
        
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return ""
            
        page = doc[page_num - 1]
        text = page.get_text()
        doc.close()
        
        return text


class PDFSplitter:
    """PDF拆分器"""
    
    def __init__(self):
        self._fitz = None
        
    def _init_fitz(self):
        """延迟初始化PyMuPDF"""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                logger.error("PyMuPDF not installed. Please run: pip install PyMuPDF")
                raise
                
    def split_pages(self, pdf_path: str, output_dir: str, 
                    page_ranges: List[Tuple[int, int, str]]) -> List[str]:
        """
        按页码范围拆分PDF
        
        Args:
            pdf_path: 源PDF路径
            output_dir: 输出目录
            page_ranges: 页码范围列表，每项为 (start, end, output_name)
                         页码从1开始
            
        Returns:
            输出文件路径列表
        """
        self._init_fitz()
        
        os.makedirs(output_dir, exist_ok=True)
        
        source_doc = self._fitz.open(pdf_path)
        output_files = []
        
        for start, end, name in page_ranges:
            # 创建新文档
            new_doc = self._fitz.open()
            
            # 复制页面（PyMuPDF页码从0开始）
            for page_num in range(start - 1, end):
                if 0 <= page_num < len(source_doc):
                    new_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)
                    
            # 保存
            output_path = os.path.join(output_dir, f"{name}.pdf")
            new_doc.save(output_path)
            new_doc.close()
            
            output_files.append(output_path)
            logger.info(f"Created: {output_path} (pages {start}-{end})")
            
        source_doc.close()
        return output_files
        
    def split_by_type(self, pdf_path: str, output_dir: str,
                      page_classifications: List[Tuple[int, str, str]]) -> Dict[str, List[str]]:
        """
        按文档类型拆分PDF
        
        Args:
            pdf_path: 源PDF路径
            output_dir: 输出目录
            page_classifications: 页面分类列表，每项为 (page_num, doc_type, doc_name)
            
        Returns:
            文档类型 -> 输出文件列表的映射
        """
        self._init_fitz()
        
        # 按类型分组连续页面
        groups = []
        current_group = None
        
        for page_num, doc_type, doc_name in sorted(page_classifications, key=lambda x: x[0]):
            if current_group is None or current_group['type'] != doc_type:
                if current_group:
                    groups.append(current_group)
                current_group = {
                    'type': doc_type,
                    'name': doc_name,
                    'start': page_num,
                    'end': page_num
                }
            else:
                current_group['end'] = page_num
                
        if current_group:
            groups.append(current_group)
            
        # 创建类型子目录并拆分
        result = {}
        
        for group in groups:
            type_dir = os.path.join(output_dir, group['type'])
            os.makedirs(type_dir, exist_ok=True)
            
            output_files = self.split_pages(
                pdf_path, type_dir,
                [(group['start'], group['end'], group['name'])]
            )
            
            if group['type'] not in result:
                result[group['type']] = []
            result[group['type']].extend(output_files)
            
        return result


class PDFMerger:
    """PDF合并器"""
    
    def __init__(self):
        self._fitz = None
        
    def _init_fitz(self):
        """延迟初始化PyMuPDF"""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                logger.error("PyMuPDF not installed. Please run: pip install PyMuPDF")
                raise
                
    def merge(self, pdf_paths: List[str], output_path: str) -> bool:
        """
        合并多个PDF文件
        
        Args:
            pdf_paths: PDF文件路径列表
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        self._init_fitz()
        
        try:
            merged_doc = self._fitz.open()
            
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    doc = self._fitz.open(pdf_path)
                    merged_doc.insert_pdf(doc)
                    doc.close()
                else:
                    logger.warning(f"PDF not found: {pdf_path}")
                    
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            merged_doc.save(output_path)
            merged_doc.close()
            
            logger.info(f"Merged {len(pdf_paths)} PDFs to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge PDFs: {e}")
            return False


class PDFCopier:
    """PDF复制器（用于按分类复制文件）"""
    
    @staticmethod
    def copy_to_category(source_path: str, output_base: str, 
                         project: str, doc_type: str, new_name: Optional[str] = None) -> str:
        """
        将PDF复制到分类目录
        
        Args:
            source_path: 源文件路径
            output_base: 输出基础目录
            project: 项目名称
            doc_type: 文档类型
            new_name: 新文件名（可选）
            
        Returns:
            输出文件路径
        """
        # 构建输出目录
        output_dir = os.path.join(output_base, project, doc_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定文件名
        if new_name:
            filename = new_name if new_name.endswith('.pdf') else f"{new_name}.pdf"
        else:
            filename = os.path.basename(source_path)
            
        output_path = os.path.join(output_dir, filename)
        
        # 处理文件名冲突
        counter = 1
        base_name = os.path.splitext(filename)[0]
        while os.path.exists(output_path):
            filename = f"{base_name}_{counter}.pdf"
            output_path = os.path.join(output_dir, filename)
            counter += 1
            
        # 复制文件
        shutil.copy2(source_path, output_path)
        logger.debug(f"Copied: {source_path} -> {output_path}")
        
        return output_path


class PDFProcessor:
    """PDF处理器（整合所有功能）"""
    
    def __init__(self, config: dict):
        """
        初始化PDF处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        processing_config = config.get('processing', {})
        
        self.dpi = processing_config.get('dpi', 200)
        self.batch_size = processing_config.get('batch_size', 10)
        
        # 初始化组件
        self.to_image = PDFToImage(dpi=self.dpi)
        self.reader = PDFReader()
        self.splitter = PDFSplitter()
        self.merger = PDFMerger()
        self.copier = PDFCopier()
        
    def process_pdf(self, pdf_path: str) -> Generator[Tuple[int, 'Image.Image'], None, None]:
        """
        处理PDF文件，逐页返回图像
        
        Args:
            pdf_path: PDF文件路径
            
        Yields:
            (页码, 图像) 元组
        """
        page_num = 1
        for image in self.to_image.convert_all_pages(pdf_path):
            yield page_num, image
            page_num += 1
            
    def get_page_count(self, pdf_path: str) -> int:
        """获取PDF页数"""
        return self.reader.get_page_count(pdf_path)
        
    def copy_to_output(self, source_path: str, project: str, doc_type: str,
                       new_name: Optional[str] = None) -> str:
        """
        复制PDF到输出目录
        
        Args:
            source_path: 源文件路径
            project: 项目名称
            doc_type: 文档类型
            new_name: 新文件名
            
        Returns:
            输出文件路径
        """
        output_base = self.config.get('paths', {}).get('output_dir', '')
        return self.copier.copy_to_category(source_path, output_base, project, doc_type, new_name)


if __name__ == "__main__":
    # 测试代码
    config = {
        'processing': {
            'dpi': 200,
            'batch_size': 10
        },
        'paths': {
            'output_dir': 'C:/temp/pdf_output'
        }
    }
    
    processor = PDFProcessor(config)
    
    # 测试PDF信息获取
    test_pdf = r"C:\Users\Kevin\Desktop\excel\致同\凭证\SARS\2003-05-31 记 30 生物孵化器SARS项目加固工程款（首期）.pdf"
    
    if os.path.exists(test_pdf):
        info = processor.reader.get_info(test_pdf)
        print(f"PDF: {info.file_path}")
        print(f"Pages: {info.page_count}")
        for page in info.pages:
            print(f"  Page {page.page_num}: {page.width}x{page.height}")
    else:
        print(f"Test PDF not found: {test_pdf}")
