"""
实时进度显示模块
使用rich库实现漂亮的控制台进度展示
"""

import time
from typing import Optional, List
from dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


console = Console()


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_files: int = 0
    processed_files: int = 0
    total_pages: int = 0
    processed_pages: int = 0
    current_file: str = ""
    current_page: int = 0
    ocr_time_total: float = 0.0
    start_time: float = 0.0
    
    @property
    def avg_ocr_time(self) -> float:
        """平均OCR时间"""
        if self.processed_pages > 0:
            return self.ocr_time_total / self.processed_pages
        return 0.0
    
    @property
    def elapsed_time(self) -> float:
        """已用时间"""
        if self.start_time > 0:
            return time.time() - self.start_time
        return 0.0
    
    @property
    def eta(self) -> float:
        """预计剩余时间"""
        if self.processed_pages > 0 and self.total_pages > 0:
            remaining = self.total_pages - self.processed_pages
            return remaining * self.avg_ocr_time
        return 0.0


class ProgressDisplay:
    """实时进度显示器"""
    
    def __init__(self):
        self.stats = ProcessingStats()
        self.progress = None
        self.live = None
        self.file_task = None
        self.page_task = None
        
    def start(self, total_files: int, total_pages: int = 0):
        """开始显示进度"""
        self.stats = ProcessingStats(
            total_files=total_files,
            total_pages=total_pages,
            start_time=time.time()
        )
        
        # 创建进度条
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        )
        
        self.file_task = self.progress.add_task(
            "[cyan]📁 Files", 
            total=total_files
        )
        
        if total_pages > 0:
            self.page_task = self.progress.add_task(
                "[green]📄 Pages", 
                total=total_pages
            )
        
        self.progress.start()
        
    def stop(self):
        """停止显示"""
        if self.progress:
            self.progress.stop()
            self._print_summary()
    
    def update_file(self, file_path: str, file_pages: int = 0):
        """更新当前处理的文件"""
        self.stats.current_file = file_path
        
        # 更新文件进度描述
        short_name = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."
        
        if self.progress and self.file_task is not None:
            self.progress.update(
                self.file_task,
                description=f"[cyan]📁 {short_name}"
            )
    
    def complete_file(self):
        """完成一个文件"""
        self.stats.processed_files += 1
        
        if self.progress and self.file_task is not None:
            self.progress.update(self.file_task, advance=1)
    
    def update_page(self, page_num: int, total_pages: int):
        """更新页面进度"""
        self.stats.current_page = page_num
        
        if self.progress and self.page_task is not None:
            self.progress.update(
                self.page_task,
                description=f"[green]📄 Page {page_num}/{total_pages}"
            )
    
    def complete_page(self, ocr_time: float = 0.0):
        """完成一页"""
        self.stats.processed_pages += 1
        self.stats.ocr_time_total += ocr_time
        
        if self.progress and self.page_task is not None:
            self.progress.update(self.page_task, advance=1)
    
    def set_total_pages(self, total: int):
        """设置总页数"""
        self.stats.total_pages = total
        
        if self.progress and self.page_task is not None:
            self.progress.update(self.page_task, total=total)
    
    def add_pages(self, count: int):
        """增加总页数"""
        self.stats.total_pages += count
        
        if self.progress:
            if self.page_task is None:
                self.page_task = self.progress.add_task(
                    "[green]📄 Pages", 
                    total=self.stats.total_pages
                )
            else:
                self.progress.update(self.page_task, total=self.stats.total_pages)
    
    def _print_summary(self):
        """打印处理摘要"""
        elapsed = self.stats.elapsed_time
        
        console.print()
        console.print(Panel.fit(
            f"[bold green]✅ Processing Complete![/]\n\n"
            f"📁 Files processed: [cyan]{self.stats.processed_files}[/] / {self.stats.total_files}\n"
            f"📄 Pages processed: [cyan]{self.stats.processed_pages}[/]\n"
            f"⏱️  Total time: [yellow]{self._format_time(elapsed)}[/]\n"
            f"📊 Avg OCR time: [yellow]{self.stats.avg_ocr_time:.2f}s[/] per page",
            title="[bold]Summary[/]",
            border_style="green"
        ))
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)}m {int(s)}s"
        else:
            h, remainder = divmod(seconds, 3600)
            m, s = divmod(remainder, 60)
            return f"{int(h)}h {int(m)}m {int(s)}s"


class SimpleProgress:
    """简单进度显示（无rich依赖）"""
    
    def __init__(self):
        self.stats = ProcessingStats()
        self.last_print_time = 0
        
    def start(self, total_files: int, total_pages: int = 0):
        """开始"""
        self.stats = ProcessingStats(
            total_files=total_files,
            total_pages=total_pages,
            start_time=time.time()
        )
        print(f"\n{'='*60}")
        print(f"Starting processing: {total_files} files")
        print(f"{'='*60}\n")
    
    def stop(self):
        """停止"""
        elapsed = self.stats.elapsed_time
        print(f"\n{'='*60}")
        print(f"✅ Complete!")
        print(f"   Files: {self.stats.processed_files}/{self.stats.total_files}")
        print(f"   Pages: {self.stats.processed_pages}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Avg: {self.stats.avg_ocr_time:.2f}s/page")
        print(f"{'='*60}\n")
    
    def update_file(self, file_path: str, file_pages: int = 0):
        """更新文件"""
        self.stats.current_file = file_path
        short_name = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
        print(f"\n📁 [{self.stats.processed_files+1}/{self.stats.total_files}] {short_name}")
    
    def complete_file(self):
        """完成文件"""
        self.stats.processed_files += 1
    
    def update_page(self, page_num: int, total_pages: int):
        """更新页面"""
        self.stats.current_page = page_num
        # 每秒最多打印一次
        now = time.time()
        if now - self.last_print_time >= 1.0:
            pct = (page_num / total_pages * 100) if total_pages > 0 else 0
            print(f"   📄 Page {page_num}/{total_pages} ({pct:.0f}%)", end='\r')
            self.last_print_time = now
    
    def complete_page(self, ocr_time: float = 0.0):
        """完成页面"""
        self.stats.processed_pages += 1
        self.stats.ocr_time_total += ocr_time
    
    def set_total_pages(self, total: int):
        """设置总页数"""
        self.stats.total_pages = total
    
    def add_pages(self, count: int):
        """增加页数"""
        self.stats.total_pages += count


def get_progress_display(use_rich: bool = True):
    """获取进度显示器"""
    if use_rich:
        try:
            from rich.console import Console
            return ProgressDisplay()
        except ImportError:
            pass
    return SimpleProgress()
