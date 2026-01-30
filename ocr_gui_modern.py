"""
PDF OCR处理工具 - 现代化GUI
==============================

使用CustomTkinter创建美观的现代界面
支持暗色/亮色主题切换
"""

import os
import sys
import threading
import queue
from pathlib import Path
from datetime import datetime
import logging

# 尝试导入customtkinter，如果没有则使用标准tkinter
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")  # 默认暗色主题
    ctk.set_default_color_theme("blue")
    USE_CUSTOM_TK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk
    USE_CUSTOM_TK = False
    print("提示: 安装 customtkinter 可获得更美观的界面")
    print("pip install customtkinter")

from tkinter import filedialog, messagebox

# 添加当前目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernOCRApp:
    """现代化OCR应用界面"""
    
    def __init__(self):
        if USE_CUSTOM_TK:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        
        self.root.title("PDF OCR 智能处理工具")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 状态变量
        self.voucher_folder = ""
        self.reference_folder = ""
        self.output_folder = ""
        self.is_running = False
        self.msg_queue = queue.Queue()
        
        # 创建界面
        self._create_ui()
        
        # 定时检查消息队列
        self.root.after(100, self._check_queue)
    
    def _create_ui(self):
        """创建用户界面"""
        if USE_CUSTOM_TK:
            self._create_modern_ui()
        else:
            self._create_classic_ui()
    
    def _create_modern_ui(self):
        """创建现代化界面（CustomTkinter）"""
        # 主容器
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ============ 标题区域 ============
        title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="📄 PDF OCR 智能处理工具",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(side="left")
        
        # 主题切换按钮
        self.theme_btn = ctk.CTkButton(
            title_frame,
            text="🌙",
            width=40,
            command=self._toggle_theme
        )
        self.theme_btn.pack(side="right")
        
        # ============ 文件夹选择区域 ============
        folder_frame = ctk.CTkFrame(main_frame)
        folder_frame.pack(fill="x", pady=10)
        
        # 凭证文件夹
        self._create_folder_row_modern(
            folder_frame, "📁 凭证文件夹", "voucher", 0
        )
        
        # 参照资料文件夹
        self._create_folder_row_modern(
            folder_frame, "📂 参照资料文件夹", "reference", 1
        )
        
        # 输出文件夹
        self._create_folder_row_modern(
            folder_frame, "📤 输出文件夹", "output", 2
        )
        
        # ============ 选项区域 ============
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(fill="x", pady=10)
        
        options_label = ctk.CTkLabel(
            options_frame,
            text="处理选项",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        options_label.pack(anchor="w", padx=10, pady=5)
        
        options_inner = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_inner.pack(fill="x", padx=10, pady=5)
        
        # OCR引擎选择
        engine_label = ctk.CTkLabel(options_inner, text="OCR引擎:")
        engine_label.pack(side="left", padx=(0, 10))
        
        self.engine_var = ctk.StringVar(value="DeepSeek-OCR2")
        engine_menu = ctk.CTkOptionMenu(
            options_inner,
            values=["DeepSeek-OCR2", "PaddleOCR"],
            variable=self.engine_var,
            width=150
        )
        engine_menu.pack(side="left", padx=(0, 30))
        
        # DPI选择
        dpi_label = ctk.CTkLabel(options_inner, text="DPI:")
        dpi_label.pack(side="left", padx=(0, 10))
        
        self.dpi_var = ctk.StringVar(value="150")
        dpi_menu = ctk.CTkOptionMenu(
            options_inner,
            values=["100", "150", "200", "300"],
            variable=self.dpi_var,
            width=100
        )
        dpi_menu.pack(side="left", padx=(0, 30))
        
        # 生成可搜索PDF
        self.searchable_var = ctk.BooleanVar(value=True)
        searchable_check = ctk.CTkCheckBox(
            options_inner,
            text="生成可搜索PDF",
            variable=self.searchable_var
        )
        searchable_check.pack(side="left")
        
        # ============ 进度区域 ============
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(fill="x", pady=10)
        
        progress_label = ctk.CTkLabel(
            progress_frame,
            text="处理进度",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        progress_label.pack(anchor="w", padx=10, pady=5)
        
        # 当前文件
        self.current_file_label = ctk.CTkLabel(
            progress_frame,
            text="等待开始...",
            font=ctk.CTkFont(size=12)
        )
        self.current_file_label.pack(anchor="w", padx=10)
        
        # 进度条
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(fill="x", padx=10, pady=10)
        self.progress_bar.set(0)
        
        # 状态标签
        self.status_label = ctk.CTkLabel(
            progress_frame,
            text="就绪",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(anchor="w", padx=10)
        
        # ============ 日志区域 ============
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.pack(fill="both", expand=True, pady=10)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="处理日志",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        log_label.pack(anchor="w", padx=10, pady=5)
        
        self.log_textbox = ctk.CTkTextbox(log_frame, height=150)
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        # ============ 按钮区域 ============
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=10)
        
        # 开始按钮
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="▶ 开始处理",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            width=200,
            command=self._start_processing
        )
        self.start_btn.pack(side="left", padx=5)
        
        # 停止按钮
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="■ 停止",
            font=ctk.CTkFont(size=16),
            height=50,
            width=100,
            fg_color="gray",
            command=self._stop_processing,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # 打开输出文件夹
        self.open_btn = ctk.CTkButton(
            button_frame,
            text="📁 打开输出文件夹",
            font=ctk.CTkFont(size=14),
            height=50,
            width=180,
            fg_color="green",
            command=self._open_output_folder
        )
        self.open_btn.pack(side="right", padx=5)
        
        # 统计信息
        self.stats_label = ctk.CTkLabel(
            button_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.stats_label.pack(side="right", padx=20)
    
    def _create_folder_row_modern(self, parent, label_text, folder_type, row):
        """创建现代化的文件夹选择行"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=8)
        
        label = ctk.CTkLabel(
            frame,
            text=label_text,
            font=ctk.CTkFont(size=13),
            width=150,
            anchor="w"
        )
        label.pack(side="left")
        
        entry = ctk.CTkEntry(frame, width=400, height=35)
        entry.pack(side="left", fill="x", expand=True, padx=10)
        
        # 保存entry引用
        setattr(self, f"{folder_type}_entry", entry)
        
        btn = ctk.CTkButton(
            frame,
            text="浏览",
            width=80,
            height=35,
            command=lambda: self._browse_folder(folder_type)
        )
        btn.pack(side="right")
    
    def _create_classic_ui(self):
        """创建经典界面（标准tkinter）"""
        # 简化版本，使用标准tkinter（作为fallback）
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # 标题
        title = ttk.Label(main_frame, text="PDF OCR 处理工具", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # 文件夹选择
        for label_text, folder_type in [
            ("凭证文件夹:", "voucher"),
            ("参照资料文件夹:", "reference"),
            ("输出文件夹:", "output")
        ]:
            frame = ttk.Frame(main_frame)
            frame.pack(fill="x", pady=5)
            
            label = ttk.Label(frame, text=label_text, width=15)
            label.pack(side="left")
            
            entry = ttk.Entry(frame)
            entry.pack(side="left", fill="x", expand=True, padx=5)
            setattr(self, f"{folder_type}_entry", entry)
            
            btn = ttk.Button(frame, text="浏览", 
                           command=lambda t=folder_type: self._browse_folder(t))
            btn.pack(side="right")
        
        # 进度条
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=10)
        
        # 状态
        self.status_label = ttk.Label(main_frame, text="就绪")
        self.status_label.pack()
        
        self.current_file_label = ttk.Label(main_frame, text="")
        self.current_file_label.pack()
        
        # 日志
        self.log_textbox = tk.Text(main_frame, height=10)
        self.log_textbox.pack(fill="both", expand=True, pady=10)
        
        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(btn_frame, text="开始处理", command=self._start_processing)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self._stop_processing, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        self.open_btn = ttk.Button(btn_frame, text="打开输出", command=self._open_output_folder)
        self.open_btn.pack(side="right", padx=5)
        
        self.stats_label = ttk.Label(btn_frame, text="")
        self.stats_label.pack(side="right", padx=10)
    
    def _toggle_theme(self):
        """切换主题"""
        if USE_CUSTOM_TK:
            current = ctk.get_appearance_mode()
            if current == "Dark":
                ctk.set_appearance_mode("light")
                self.theme_btn.configure(text="☀️")
            else:
                ctk.set_appearance_mode("dark")
                self.theme_btn.configure(text="🌙")
    
    def _browse_folder(self, folder_type):
        """浏览文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            entry = getattr(self, f"{folder_type}_entry")
            if USE_CUSTOM_TK:
                entry.delete(0, "end")
                entry.insert(0, folder)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, folder)
            setattr(self, f"{folder_type}_folder", folder)
    
    def _log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        
        if USE_CUSTOM_TK:
            self.log_textbox.insert("end", log_msg)
            self.log_textbox.see("end")
        else:
            self.log_textbox.insert(tk.END, log_msg)
            self.log_textbox.see(tk.END)
    
    def _check_queue(self):
        """检查消息队列"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg.get('type')
                
                if msg_type == 'log':
                    self._log(msg['text'])
                elif msg_type == 'progress':
                    if USE_CUSTOM_TK:
                        self.progress_bar.set(msg['value'] / 100)
                    else:
                        self.progress_bar['value'] = msg['value']
                elif msg_type == 'status':
                    if USE_CUSTOM_TK:
                        self.status_label.configure(text=msg['text'])
                    else:
                        self.status_label.configure(text=msg['text'])
                elif msg_type == 'file':
                    if USE_CUSTOM_TK:
                        self.current_file_label.configure(text=f"当前: {msg['text']}")
                    else:
                        self.current_file_label.configure(text=f"当前: {msg['text']}")
                elif msg_type == 'done':
                    self._processing_done(msg.get('success', True), msg.get('stats'))
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self._check_queue)
    
    def _start_processing(self):
        """开始处理"""
        # 获取路径
        self.voucher_folder = self.voucher_entry.get()
        self.reference_folder = self.reference_entry.get()
        self.output_folder = self.output_entry.get()
        
        # 验证
        if not self.voucher_folder or not os.path.isdir(self.voucher_folder):
            messagebox.showerror("错误", "请选择有效的凭证文件夹")
            return
        
        if not self.reference_folder or not os.path.isdir(self.reference_folder):
            messagebox.showerror("错误", "请选择有效的参照资料文件夹")
            return
        
        # 默认输出目录
        if not self.output_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = str(Path.home() / "Desktop" / f"OCR_结果_{timestamp}")
            if USE_CUSTOM_TK:
                self.output_entry.delete(0, "end")
                self.output_entry.insert(0, self.output_folder)
            else:
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, self.output_folder)
        
        # 更新UI
        self.is_running = True
        if USE_CUSTOM_TK:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.progress_bar.set(0)
        else:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.progress_bar['value'] = 0
        
        self._log("开始处理...")
        
        # 启动处理线程
        thread = threading.Thread(target=self._run_processing, daemon=True)
        thread.start()
    
    def _run_processing(self):
        """后台处理线程"""
        try:
            from run_ocr import run_ocr_pipeline_with_callback
            
            def callback(msg_type, **kwargs):
                if not self.is_running:
                    raise InterruptedError("用户取消")
                self.msg_queue.put({'type': msg_type, **kwargs})
            
            stats = run_ocr_pipeline_with_callback(
                self.voucher_folder,
                self.reference_folder,
                self.output_folder,
                callback
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
        self._log("正在停止...")
    
    def _processing_done(self, success, stats=None):
        """处理完成"""
        self.is_running = False
        
        if USE_CUSTOM_TK:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
        
        if success and stats:
            if USE_CUSTOM_TK:
                self.progress_bar.set(1)
                self.status_label.configure(text="✅ 处理完成!")
                self.stats_label.configure(
                    text=f"文件: {stats.get('voucher_files', 0)}+{stats.get('reference_files', 0)} | "
                         f"匹配: {stats.get('matched', 0)}"
                )
            else:
                self.progress_bar['value'] = 100
                self.status_label.configure(text="处理完成!")
            
            self._log("=" * 50)
            self._log(f"凭证: {stats.get('voucher_files', 0)} 文件, {stats.get('voucher_pages', 0)} 页")
            self._log(f"参照: {stats.get('reference_files', 0)} 文件, {stats.get('reference_pages', 0)} 页")
            self._log(f"匹配: {stats.get('matched', 0)} | 部分: {stats.get('partial', 0)} | 未匹配: {stats.get('unmatched', 0)}")
            self._log(f"输出: {stats.get('output_folder', '')}")
            
            messagebox.showinfo("完成", 
                f"处理完成!\n\n"
                f"凭证: {stats.get('voucher_files', 0)} 文件\n"
                f"参照: {stats.get('reference_files', 0)} 文件\n\n"
                f"匹配: {stats.get('matched', 0)} 页"
            )
        else:
            if USE_CUSTOM_TK:
                self.status_label.configure(text="已停止")
            else:
                self.status_label.configure(text="已停止")
    
    def _open_output_folder(self):
        """打开输出文件夹"""
        folder = self.output_entry.get() if hasattr(self, 'output_entry') else self.output_folder
        if folder and os.path.isdir(folder):
            os.startfile(folder)
        else:
            messagebox.showwarning("提示", "输出文件夹不存在")
    
    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    app = ModernOCRApp()
    app.run()


if __name__ == "__main__":
    main()
