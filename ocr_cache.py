"""
OCR结果缓存模块
缓存OCR结果到磁盘，避免重复处理同一文件
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class OCRCache:
    """OCR结果缓存管理器"""
    
    def __init__(self, cache_dir: str):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self._index = self._load_index()
        
    def _load_index(self) -> Dict:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
        
    def _save_index(self):
        """保存缓存索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
            
    def _get_file_hash(self, file_path: str) -> str:
        """获取文件的MD5哈希值（用于检测文件变化）"""
        hasher = hashlib.md5()
        stat = os.stat(file_path)
        # 使用文件路径+大小+修改时间作为快速哈希
        key = f"{file_path}|{stat.st_size}|{stat.st_mtime}"
        hasher.update(key.encode('utf-8'))
        return hasher.hexdigest()
        
    def _get_cache_path(self, file_hash: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{file_hash}.json"
        
    def get(self, file_path: str) -> Optional[List[Dict]]:
        """
        从缓存获取OCR结果
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            缓存的OCR结果，如果没有缓存则返回None
        """
        file_hash = self._get_file_hash(file_path)
        
        # 检查索引
        if file_hash not in self._index:
            return None
            
        cache_path = self._get_cache_path(file_hash)
        
        if not cache_path.exists():
            # 缓存文件不存在，清理索引
            del self._index[file_hash]
            self._save_index()
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Cache hit: {file_path}")
                return data['pages']
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
            
    def put(self, file_path: str, page_features: List) -> bool:
        """
        将OCR结果存入缓存
        
        Args:
            file_path: PDF文件路径
            page_features: 页面特征列表
            
        Returns:
            是否成功
        """
        file_hash = self._get_file_hash(file_path)
        cache_path = self._get_cache_path(file_hash)
        
        try:
            # 序列化页面特征
            pages_data = []
            for pf in page_features:
                pages_data.append({
                    'file_path': pf.file_path,
                    'page_num': pf.page_num,
                    'text': pf.text,
                    'doc_type': pf.doc_type,
                    'dates': pf.dates,
                    'amounts': pf.amounts,
                    'numbers': pf.numbers,
                    'keywords': pf.keywords if hasattr(pf, 'keywords') else []
                })
                
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'source_file': file_path,
                    'pages': pages_data
                }, f, ensure_ascii=False, indent=2)
                
            # 更新索引
            self._index[file_hash] = {
                'file': file_path,
                'pages': len(pages_data)
            }
            self._save_index()
            
            logger.debug(f"Cached: {file_path} ({len(pages_data)} pages)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")
            return False
            
    def clear(self):
        """清除所有缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        logger.info("Cache cleared")


def restore_page_features(cached_data: List[Dict]):
    """
    从缓存数据恢复PageFeatures对象
    
    Args:
        cached_data: 缓存的页面数据列表
        
    Returns:
        PageFeatures对象列表
    """
    from content_matcher import PageFeatures
    
    features = []
    for data in cached_data:
        pf = PageFeatures(
            file_path=data['file_path'],
            page_num=data['page_num'],
            text=data['text'],
            doc_type=data['doc_type'],
            dates=data.get('dates', []),
            amounts=data.get('amounts', []),
            numbers=data.get('numbers', []),
            keywords=data.get('keywords', [])
        )
        features.append(pf)
    return features
