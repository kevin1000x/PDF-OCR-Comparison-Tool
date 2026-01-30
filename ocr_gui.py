"""
PDF OCR处理工具 - 桌面GUI版
============================

功能：
1. 图形界面选择文件夹
2. 实时显示处理进度
3. 生成可搜索PDF和对比报告
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import queue
import logging

# 添加当前目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRApplication:
    """PDF OCR处理工具 GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PDF OCR处理工具")
        self.root.geometry("700x550")
        self.root.resizable(True, True)
        
        # 设置图标（如果有）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # 状态变量
        self.voucher_folder = tk.StringVar()
        self.reference_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.status_text = tk.StringVar(value="就绪")
        self.progress_value = tk.DoubleVar(value=0)
        self.current_file = tk.StringVar(value="")
        
        # 消息队列（用于线程间通信）
        self.msg_queue = queue.Queue()
        
        # 处理线程
        self.processing_thread = None
        self.is_running = False
        
        # 创建界面
        self._create_widgets()
        
        # 定时检查消息队列
        self.root.after(100, self._check_queue)
    
    def _create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(
            main_frame, 
            text="PDF OCR处理工具", 
            font=("Microsoft YaHei", 16, "bold")
        )
        title_label.pack(pady=(0, 15))
        
        # 文件夹选择框架
        folder_frame = ttk.LabelFrame(main_frame, text="文件夹设置", padding="10")
        folder_frame.pack(fill=tk.X, pady=5)
        
        # 凭证文件夹
        self._create_folder_row(
            folder_frame, 
            "凭证文件夹:", 
            self.voucher_folder, 
            0
        )
        
        # 参照资料文件夹
        self._create_folder_row(
            folder_frame, 
            "参照资料文件夹:", 
            self.reference_folder, 
            1
        )
        
        # 输出文件夹
        self._create_folder_row(
            folder_frame, 
            "输出文件夹:", 
            self.output_folder, 
            2
        )
        
        # 进度框架
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        # 当前文件
        file_label = ttk.Label(progress_frame, textvariable=self.current_file)
        file_label.pack(fill=tk.X)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_value,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 状态文本
        status_label = ttk.Label(
            progress_frame, 
            textvariable=self.status_text,
            font=("Microsoft YaHei", 10)
        )
        status_label.pack(fill=tk.X)
        
        # 日志框架
        log_frame = ttk.LabelFrame(main_frame, text="处理日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 开始按钮
        self.start_button = ttk.Button(
            button_frame, 
            text="▶ 开始处理", 
            command=self._start_processing,
            width=20
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # 停止按钮
        self.stop_button = ttk.Button(
            button_frame, 
            text="■ 停止", 
            command=self._stop_processing,
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 打开输出文件夹按钮
        self.open_button = ttk.Button(
            button_frame, 
            text="📁 打开输出文件夹", 
            command=self._open_output_folder,
            width=18
        )
        self.open_button.pack(side=tk.RIGHT, padx=5)
    
    def _create_folder_row(self, parent, label_text, variable, row):
        """创建文件夹选择行"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        
        label = ttk.Label(frame, text=label_text, width=15)
        label.pack(side=tk.LEFT)
        
        entry = ttk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        button = ttk.Button(
            frame, 
            text="浏览...", 
            command=lambda: self._browse_folder(variable),
            width=8
        )
        button.pack(side=tk.RIGHT)
    
    def _browse_folder(self, variable):
        """浏览文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)
    
    def _log(self, message):
        """添加日志"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _check_queue(self):
        """检查消息队列"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg.get('type')
                
                if msg_type == 'log':
                    self._log(msg['text'])
                elif msg_type == 'progress':
                    self.progress_value.set(msg['value'])
                elif msg_type == 'status':
                    self.status_text.set(msg['text'])
                elif msg_type == 'file':
                    self.current_file.set(f"当前文件: {msg['text']}")
                elif msg_type == 'done':
                    self._processing_done(msg.get('success', True), msg.get('stats'))
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self._check_queue)
    
    def _start_processing(self):
        """开始处理"""
        # 验证输入
        if not self.voucher_folder.get():
            messagebox.showerror("错误", "请选择凭证文件夹")
            return
        
        if not self.reference_folder.get():
            messagebox.showerror("错误", "请选择参照资料文件夹")
            return
        
        if not os.path.isdir(self.voucher_folder.get()):
            messagebox.showerror("错误", "凭证文件夹不存在")
            return
        
        if not os.path.isdir(self.reference_folder.get()):
            messagebox.showerror("错误", "参照资料文件夹不存在")
            return
        
        # 设置默认输出文件夹
        if not self.output_folder.get():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_output = str(Path.home() / "Desktop" / f"OCR_结果_{timestamp}")
            self.output_folder.set(default_output)
        
        # 更新UI
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_value.set(0)
        self.status_text.set("正在初始化...")
        
        # 清空日志
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 启动处理线程
        self.processing_thread = threading.Thread(
            target=self._run_processing,
            daemon=True
        )
        self.processing_thread.start()
        
        self._log("处理开始...")
    
    def _run_processing(self):
        """在后台线程中运行处理"""
        try:
            from run_ocr import run_ocr_pipeline_with_callback
            
            def progress_callback(msg_type, **kwargs):
                if not self.is_running:
                    raise InterruptedError("用户取消")
                self.msg_queue.put({'type': msg_type, **kwargs})
            
            stats = run_ocr_pipeline_with_callback(
                self.voucher_folder.get(),
                self.reference_folder.get(),
                self.output_folder.get(),
                progress_callback
            )
            
            self.msg_queue.put({'type': 'done', 'success': True, 'stats': stats})
            
        except InterruptedError:
            self.msg_queue.put({'type': 'log', 'text': '处理已取消'})
            self.msg_queue.put({'type': 'done', 'success': False})
        except Exception as e:
            self.msg_queue.put({'type': 'log', 'text': f'错误: {e}'})
            self.msg_queue.put({'type': 'done', 'success': False})
    
    def _stop_processing(self):
        """停止处理"""
        self.is_running = False
        self.status_text.set("正在停止...")
        self._log("正在停止处理...")
    
    def _processing_done(self, success, stats=None):
        """处理完成"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if success and stats:
            self.progress_value.set(100)
            self.status_text.set("✅ 处理完成!")
            self._log("="*50)
            self._log(f"凭证文件: {stats.get('voucher_files', 0)} 个 ({stats.get('voucher_pages', 0)} 页)")
            self._log(f"参照文件: {stats.get('reference_files', 0)} 个 ({stats.get('reference_pages', 0)} 页)")
            self._log(f"匹配: {stats.get('matched', 0)} / 部分匹配: {stats.get('partial', 0)} / 未匹配: {stats.get('unmatched', 0)}")
            self._log(f"输出目录: {stats.get('output_folder', '')}")
            self._log("="*50)
            
            messagebox.showinfo("完成", 
                f"处理完成!\n\n"
                f"凭证: {stats.get('voucher_files', 0)} 个文件\n"
                f"参照: {stats.get('reference_files', 0)} 个文件\n\n"
                f"输出目录:\n{stats.get('output_folder', '')}"
            )
        else:
            self.status_text.set("已停止")
    
    def _open_output_folder(self):
        """打开输出文件夹"""
        folder = self.output_folder.get()
        if folder and os.path.isdir(folder):
            os.startfile(folder)
        else:
            messagebox.showwarning("提示", "输出文件夹不存在")


def main():
    root = tk.Tk()
    app = OCRApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
