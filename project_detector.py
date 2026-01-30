"""
项目检测器模块
根据文件路径和内容关键词识别文档所属项目
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProjectMatch:
    """项目匹配结果"""
    project_name: str  # 项目名称
    confidence: float  # 匹配置信度 (0-1)
    match_type: str  # 匹配方式: 'path' 或 'content'
    matched_keywords: List[str] = field(default_factory=list)  # 匹配的关键词


class ProjectDetector:
    """项目检测器"""
    
    def __init__(self, config: dict):
        """
        初始化项目检测器
        
        Args:
            config: 包含projects配置的字典
        """
        self.projects = config.get('projects', {})
        self._build_keyword_index()
        
    def _build_keyword_index(self):
        """构建关键词索引，用于快速匹配"""
        self.keyword_to_project = {}
        for project_name, project_config in self.projects.items():
            keywords = project_config.get('keywords', [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_to_project:
                    self.keyword_to_project[keyword_lower] = []
                self.keyword_to_project[keyword_lower].append(project_name)
                
    def detect_by_path(self, file_path: str) -> Optional[ProjectMatch]:
        """
        根据文件路径检测项目
        
        Args:
            file_path: 文件路径
            
        Returns:
            项目匹配结果，未匹配返回None
        """
        path = Path(file_path)
        path_parts = [p.lower() for p in path.parts]
        
        for project_name, project_config in self.projects.items():
            voucher_folders = project_config.get('voucher_folders', [])
            reference_folders = project_config.get('reference_folders', [])
            all_folders = voucher_folders + reference_folders
            
            for folder in all_folders:
                folder_lower = folder.lower()
                # 支持通配符匹配
                if '*' in folder:
                    pattern = folder_lower.replace('*', '.*')
                    for part in path_parts:
                        if re.match(pattern, part):
                            return ProjectMatch(
                                project_name=project_name,
                                confidence=1.0,
                                match_type='path'
                            )
                else:
                    if folder_lower in path_parts:
                        return ProjectMatch(
                            project_name=project_name,
                            confidence=1.0,
                            match_type='path'
                        )
                        
        return None
        
    def detect_by_content(self, text: str, top_n: int = 3) -> List[ProjectMatch]:
        """
        根据文本内容检测项目
        
        Args:
            text: OCR识别的文本内容
            top_n: 返回前N个最可能的项目
            
        Returns:
            项目匹配结果列表，按置信度降序排列
        """
        text_lower = text.lower()
        project_scores = {}
        project_keywords = {}
        
        for project_name, project_config in self.projects.items():
            keywords = project_config.get('keywords', [])
            matched = []
            score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # 计算关键词出现次数
                count = text_lower.count(keyword_lower)
                if count > 0:
                    matched.append(keyword)
                    # 根据关键词长度和出现次数加权
                    score += count * len(keyword)
                    
            if score > 0:
                project_scores[project_name] = score
                project_keywords[project_name] = matched
                
        if not project_scores:
            return []
            
        # 归一化得分
        max_score = max(project_scores.values())
        
        results = []
        for project_name, score in sorted(project_scores.items(), key=lambda x: -x[1])[:top_n]:
            results.append(ProjectMatch(
                project_name=project_name,
                confidence=score / max_score,
                match_type='content',
                matched_keywords=project_keywords[project_name]
            ))
            
        return results
        
    def detect(self, file_path: str, text: str = "") -> ProjectMatch:
        """
        综合检测项目归属
        
        优先使用路径匹配，如果路径无法匹配则使用内容匹配
        
        Args:
            file_path: 文件路径
            text: OCR识别的文本内容（可选）
            
        Returns:
            项目匹配结果
        """
        # 首先尝试路径匹配
        path_match = self.detect_by_path(file_path)
        if path_match:
            logger.debug(f"Path match: {file_path} -> {path_match.project_name}")
            return path_match
            
        # 如果有文本内容，尝试内容匹配
        if text:
            content_matches = self.detect_by_content(text, top_n=1)
            if content_matches:
                logger.debug(f"Content match: {file_path} -> {content_matches[0].project_name}")
                return content_matches[0]
                
        # 无法匹配，返回"未分类"
        return ProjectMatch(
            project_name="未分类",
            confidence=0.0,
            match_type='none'
        )
        
    def get_voucher_folders(self, project_name: str) -> List[str]:
        """获取项目对应的凭证文件夹列表"""
        if project_name in self.projects:
            return self.projects[project_name].get('voucher_folders', [])
        return []
        
    def get_reference_folders(self, project_name: str) -> List[str]:
        """获取项目对应的参照资料文件夹列表"""
        if project_name in self.projects:
            return self.projects[project_name].get('reference_folders', [])
        return []
        
    def get_all_projects(self) -> List[str]:
        """获取所有项目名称"""
        return list(self.projects.keys())
        
    def get_projects_with_vouchers(self) -> List[str]:
        """获取有凭证文件夹的项目"""
        return [
            name for name, config in self.projects.items()
            if config.get('voucher_folders')
        ]
        
    def get_projects_with_references(self) -> List[str]:
        """获取有参照资料的项目"""
        return [
            name for name, config in self.projects.items()
            if config.get('reference_folders')
        ]


class ProjectFolderMapper:
    """项目文件夹映射器"""
    
    def __init__(self, detector: ProjectDetector, base_paths: dict):
        """
        初始化映射器
        
        Args:
            detector: 项目检测器实例
            base_paths: 基础路径配置，包含input_vouchers和reference_docs
        """
        self.detector = detector
        self.voucher_base = Path(base_paths.get('input_vouchers', ''))
        self.reference_base = Path(base_paths.get('reference_docs', ''))
        self.output_base = Path(base_paths.get('output_dir', ''))
        
    def get_voucher_path(self, project_name: str) -> Optional[Path]:
        """获取项目的凭证文件夹路径"""
        folders = self.detector.get_voucher_folders(project_name)
        for folder in folders:
            # 处理通配符
            if '*' in folder:
                pattern = folder.replace('*', '')
                for item in self.voucher_base.iterdir():
                    if item.is_dir() and pattern in item.name:
                        return item
            else:
                path = self.voucher_base / folder
                if path.exists():
                    return path
        return None
        
    def get_reference_path(self, project_name: str) -> Optional[Path]:
        """获取项目的参照资料文件夹路径"""
        folders = self.detector.get_reference_folders(project_name)
        for folder in folders:
            path = self.reference_base / folder
            if path.exists():
                return path
        return None
        
    def get_output_path(self, project_name: str) -> Path:
        """获取项目的输出文件夹路径"""
        return self.output_base / project_name
        
    def scan_all_voucher_files(self) -> Dict[str, List[Path]]:
        """
        扫描所有凭证文件，按项目分组
        
        Returns:
            项目名 -> 文件路径列表的映射
        """
        result = {}
        
        if not self.voucher_base.exists():
            logger.warning(f"Voucher base path not found: {self.voucher_base}")
            return result
            
        for pdf_file in self.voucher_base.rglob("*.pdf"):
            match = self.detector.detect_by_path(str(pdf_file))
            if match:
                project_name = match.project_name
            else:
                project_name = "未分类"
                
            if project_name not in result:
                result[project_name] = []
            result[project_name].append(pdf_file)
            
        return result
        
    def scan_all_reference_files(self) -> Dict[str, List[Path]]:
        """
        扫描所有参照资料文件，按项目分组
        
        Returns:
            项目名 -> 文件路径列表的映射
        """
        result = {}
        
        if not self.reference_base.exists():
            logger.warning(f"Reference base path not found: {self.reference_base}")
            return result
            
        for pdf_file in self.reference_base.rglob("*.pdf"):
            match = self.detector.detect_by_path(str(pdf_file))
            if match:
                project_name = match.project_name
            else:
                project_name = "未分类"
                
            if project_name not in result:
                result[project_name] = []
            result[project_name].append(pdf_file)
            
        return result


if __name__ == "__main__":
    # 测试代码
    import yaml
    
    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        detector = ProjectDetector(config)
        
        # 测试路径匹配
        test_path = r"C:\Users\Kevin\Desktop\excel\致同\凭证\SARS\test.pdf"
        result = detector.detect_by_path(test_path)
        if result:
            print(f"Path match: {result.project_name} (confidence: {result.confidence})")
            
        # 测试内容匹配
        test_text = "生物孵化器SARS项目加固工程款，消防工程首期付款"
        results = detector.detect_by_content(test_text)
        for r in results:
            print(f"Content match: {r.project_name} (confidence: {r.confidence:.2f}, keywords: {r.matched_keywords})")
    else:
        print("Config file not found. Please create config.yaml first.")
