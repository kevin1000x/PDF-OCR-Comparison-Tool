"""
批量处理模块 - 支持多进程并行处理PDF
==========================================

提供高效的多进程PDF处理能力，显著提升大批量文件的处理速度
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """批处理结果"""
    file_path: str
    success: bool
    pages_processed: int
    processing_time: float
    error_message: Optional[str] = None
    ocr_text: Optional[str] = None


@dataclass
class BatchProgress:
    """批处理进度"""
    total_files: int
    completed_files: int
    current_file: str
    total_pages: int
    completed_pages: int
    elapsed_time: float
    estimated_remaining: float


class SmartScheduler:
    """
    智能调度器 - 根据GPU显存动态调整处理策略
    
    监控GPU资源使用情况，自动调整并发数和批处理大小
    """
    
    def __init__(self, gpu_threshold: float = 0.8, min_batch: int = 1, max_batch: int = 5):
        self.gpu_threshold = gpu_threshold  # GPU显存使用率阈值
        self.min_batch = min_batch
        self.max_batch = max_batch
        self._last_check = 0
        self._check_interval = 5  # 秒
    
    def get_gpu_memory_usage(self) -> float:
        """获取GPU显存使用率"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return allocated / total
        except:
            pass
        
        # 备选: 使用nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                return used / total
        except:
            pass
        
        return 0.5  # 默认返回中等使用率
    
    def get_optimal_batch_size(self) -> int:
        """根据GPU显存动态调整批处理大小"""
        mem_usage = self.get_gpu_memory_usage()
        
        if mem_usage > self.gpu_threshold:
            logger.debug(f"GPU memory high ({mem_usage:.1%}), using min batch size")
            return self.min_batch
        elif mem_usage > 0.6:
            return 2
        else:
            return self.max_batch
    
    def should_pause(self) -> bool:
        """判断是否应该暂停等待GPU资源"""
        return self.get_gpu_memory_usage() > 0.95


class CheckpointManager:
    """
    断点续传管理器
    
    保存处理进度，支持中断后继续处理
    """
    
    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path.cwd() / ".ocr_checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
    
    def save_progress(self, processed_files: List[str], stats: Dict = None):
        """保存处理进度"""
        import json
        data = {
            'processed_files': processed_files,
            'stats': stats or {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Checkpoint saved: {len(processed_files)} files processed")
    
    def load_progress(self) -> Tuple[List[str], Dict]:
        """加载处理进度"""
        import json
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                processed = data.get('processed_files', [])
                stats = data.get('stats', {})
                logger.info(f"Checkpoint loaded: {len(processed)} files already processed")
                return processed, stats
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return [], {}
    
    def clear(self):
        """清除检查点"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")
    
    def get_remaining_files(self, all_files: List[Path], processed: List[str]) -> List[Path]:
        """获取尚未处理的文件"""
        processed_set = set(processed)
        remaining = [f for f in all_files if str(f) not in processed_set]
        logger.info(f"Remaining files: {len(remaining)}/{len(all_files)}")
        return remaining


class BatchProcessor:
    """
    批量处理器 - 支持多进程/多线程并行处理
    
    针对OCR处理的特点进行优化：
    - GPU操作在主进程中串行执行（避免显存竞争）
    - CPU密集型操作（PDF转图像、结果解析）可以并行
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_workers = self.config.get('max_workers', 4)
        self.batch_size = self.config.get('batch_size', 10)
        self.use_threading = self.config.get('use_threading', True)  # OCR用线程更安全
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'completed_files': 0,
            'failed_files': 0,
            'total_pages': 0,
            'total_time': 0
        }
        
        # 进度回调
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[BatchProgress], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _notify_progress(self, current_file: str = ""):
        """通知进度更新"""
        if self.progress_callback:
            elapsed = time.time() - self._start_time if hasattr(self, '_start_time') else 0
            
            # 估算剩余时间
            if self.stats['completed_files'] > 0:
                avg_time = elapsed / self.stats['completed_files']
                remaining = (self.stats['total_files'] - self.stats['completed_files']) * avg_time
            else:
                remaining = 0
            
            progress = BatchProgress(
                total_files=self.stats['total_files'],
                completed_files=self.stats['completed_files'],
                current_file=current_file,
                total_pages=self.stats['total_pages'],
                completed_pages=self.stats['total_pages'],  # 简化处理
                elapsed_time=elapsed,
                estimated_remaining=remaining
            )
            self.progress_callback(progress)
    
    def process_pdf_batch(
        self,
        pdf_files: List[Path],
        ocr_engine,
        pdf_processor,
        output_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """
        批量处理PDF文件
        
        Args:
            pdf_files: PDF文件列表
            ocr_engine: OCR引擎实例
            pdf_processor: PDF处理器实例
            output_callback: 每个文件处理完成的回调
            
        Returns:
            处理结果列表
        """
        self._start_time = time.time()
        self.stats['total_files'] = len(pdf_files)
        self.stats['completed_files'] = 0
        self.stats['failed_files'] = 0
        self.stats['total_pages'] = 0
        
        results: List[BatchResult] = []
        
        logger.info(f"开始批量处理 {len(pdf_files)} 个PDF文件")
        
        # 由于GPU OCR需要串行执行，这里使用预加载策略
        # 1. 使用线程池并行进行PDF转图像
        # 2. 主线程串行执行OCR
        
        if self.use_threading and len(pdf_files) > 1:
            results = self._process_with_prefetch(
                pdf_files, ocr_engine, pdf_processor, output_callback
            )
        else:
            results = self._process_sequential(
                pdf_files, ocr_engine, pdf_processor, output_callback
            )
        
        self.stats['total_time'] = time.time() - self._start_time
        
        logger.info(f"批量处理完成: {self.stats['completed_files']}/{self.stats['total_files']} 成功")
        logger.info(f"总耗时: {self.stats['total_time']:.1f}s, 共 {self.stats['total_pages']} 页")
        
        return results
    
    def _process_sequential(
        self,
        pdf_files: List[Path],
        ocr_engine,
        pdf_processor,
        output_callback: Optional[Callable]
    ) -> List[BatchResult]:
        """串行处理（基准模式）"""
        results = []
        
        for pdf_file in pdf_files:
            self._notify_progress(pdf_file.name)
            
            result = self._process_single_pdf(
                pdf_file, ocr_engine, pdf_processor
            )
            results.append(result)
            
            if result.success:
                self.stats['completed_files'] += 1
                self.stats['total_pages'] += result.pages_processed
            else:
                self.stats['failed_files'] += 1
            
            if output_callback:
                output_callback(result)
            
            self._notify_progress(pdf_file.name)
        
        return results
    
    def _process_with_prefetch(
        self,
        pdf_files: List[Path],
        ocr_engine,
        pdf_processor,
        output_callback: Optional[Callable]
    ) -> List[BatchResult]:
        """
        预加载模式 - 在OCR处理当前文件时，预先加载下一个文件
        
        这样可以隐藏PDF转图像的延迟
        """
        results = []
        
        # 预加载队列
        prefetch_queue = Queue(maxsize=2)
        prefetch_stop = threading.Event()
        
        def prefetch_worker():
            """预加载工作线程"""
            for pdf_file in pdf_files:
                if prefetch_stop.is_set():
                    break
                try:
                    # 将PDF转换为图像
                    pages = list(pdf_processor.process_pdf(str(pdf_file)))
                    prefetch_queue.put((pdf_file, pages, None))
                except Exception as e:
                    prefetch_queue.put((pdf_file, None, str(e)))
            
            # 发送结束信号
            prefetch_queue.put(None)
        
        # 启动预加载线程
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        try:
            while True:
                item = prefetch_queue.get()
                if item is None:
                    break
                
                pdf_file, pages, error = item
                self._notify_progress(pdf_file.name)
                
                if error:
                    result = BatchResult(
                        file_path=str(pdf_file),
                        success=False,
                        pages_processed=0,
                        processing_time=0,
                        error_message=error
                    )
                else:
                    start_time = time.time()
                    try:
                        all_text = []
                        for page_num, page_image in pages:
                            ocr_result = ocr_engine.recognize_pdf_page(page_image, page_num)
                            page_text = ocr_result.get_full_text()
                            all_text.append(f"=== 第{page_num}页 ===\n{page_text}")
                        
                        result = BatchResult(
                            file_path=str(pdf_file),
                            success=True,
                            pages_processed=len(pages),
                            processing_time=time.time() - start_time,
                            ocr_text="\n\n".join(all_text)
                        )
                    except Exception as e:
                        result = BatchResult(
                            file_path=str(pdf_file),
                            success=False,
                            pages_processed=0,
                            processing_time=time.time() - start_time,
                            error_message=str(e)
                        )
                
                results.append(result)
                
                if result.success:
                    self.stats['completed_files'] += 1
                    self.stats['total_pages'] += result.pages_processed
                else:
                    self.stats['failed_files'] += 1
                
                if output_callback:
                    output_callback(result)
                
                self._notify_progress(pdf_file.name)
                
        finally:
            prefetch_stop.set()
            prefetch_thread.join(timeout=5)
        
        return results
    
    def _process_single_pdf(
        self,
        pdf_file: Path,
        ocr_engine,
        pdf_processor
    ) -> BatchResult:
        """处理单个PDF文件"""
        start_time = time.time()
        
        try:
            all_text = []
            page_count = 0
            
            for page_num, page_image in pdf_processor.process_pdf(str(pdf_file)):
                ocr_result = ocr_engine.recognize_pdf_page(page_image, page_num)
                page_text = ocr_result.get_full_text()
                all_text.append(f"=== 第{page_num}页 ===\n{page_text}")
                page_count += 1
            
            return BatchResult(
                file_path=str(pdf_file),
                success=True,
                pages_processed=page_count,
                processing_time=time.time() - start_time,
                ocr_text="\n\n".join(all_text)
            )
            
        except Exception as e:
            logger.error(f"处理失败 {pdf_file}: {e}")
            return BatchResult(
                file_path=str(pdf_file),
                success=False,
                pages_processed=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )


class PDFImagePreloader:
    """
    PDF图像预加载器
    
    在后台线程中预先将PDF转换为图像，减少等待时间
    """
    
    def __init__(self, pdf_processor, buffer_size: int = 3):
        self.pdf_processor = pdf_processor
        self.buffer_size = buffer_size
        self._buffer = Queue(maxsize=buffer_size)
        self._thread = None
        self._stop_event = threading.Event()
    
    def start(self, pdf_files: List[Path]):
        """开始预加载"""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._preload_worker,
            args=(pdf_files,),
            daemon=True
        )
        self._thread.start()
    
    def get_next(self) -> Optional[Tuple[Path, List]]:
        """获取下一个预加载的PDF"""
        try:
            return self._buffer.get(timeout=300)  # 5分钟超时
        except:
            return None
    
    def stop(self):
        """停止预加载"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _preload_worker(self, pdf_files: List[Path]):
        """预加载工作线程"""
        for pdf_file in pdf_files:
            if self._stop_event.is_set():
                break
            
            try:
                pages = list(self.pdf_processor.process_pdf(str(pdf_file)))
                self._buffer.put((pdf_file, pages))
            except Exception as e:
                logger.error(f"预加载失败 {pdf_file}: {e}")
                self._buffer.put((pdf_file, []))
        
        # 结束标记
        self._buffer.put(None)


def benchmark_processing_modes(
    pdf_files: List[Path],
    ocr_engine,
    pdf_processor,
    max_files: int = 5
) -> Dict[str, float]:
    """
    基准测试不同处理模式的性能
    
    Args:
        pdf_files: 测试文件列表
        ocr_engine: OCR引擎
        pdf_processor: PDF处理器
        max_files: 最大测试文件数
        
    Returns:
        各模式的耗时
    """
    test_files = pdf_files[:max_files]
    results = {}
    
    # 测试串行模式
    processor = BatchProcessor({'use_threading': False})
    start = time.time()
    processor.process_pdf_batch(test_files, ocr_engine, pdf_processor)
    results['sequential'] = time.time() - start
    
    # 测试预加载模式
    processor = BatchProcessor({'use_threading': True})
    start = time.time()
    processor.process_pdf_batch(test_files, ocr_engine, pdf_processor)
    results['prefetch'] = time.time() - start
    
    logger.info(f"基准测试结果: 串行={results['sequential']:.1f}s, 预加载={results['prefetch']:.1f}s")
    
    speedup = results['sequential'] / results['prefetch'] if results['prefetch'] > 0 else 1
    logger.info(f"预加载模式加速比: {speedup:.2f}x")
    
    return results
