"""
PDF OCR处理工具 - 现代化GUI v2.0
=================================

使用CustomTkinter创建美观的现代界面
- 侧边栏+Tab视图布局
- 实时状态栏
- 拖拽文件支持(可选)
"""

import os
import sys
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
import logging

# 尝试导入customtkinter
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    USE_CUSTOM_TK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk
    USE_CUSTOM_TK = False
    print("提示: 安装 customtkinter 可获得更美观的界面")
    print("pip install customtkinter")

from tkinter import filedialog, messagebox

# 尝试导入拖拽支持
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    USE_DND = True
except ImportError:
    USE_DND = False

# 添加当前目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatusBar(ctk.CTkFrame if USE_CUSTOM_TK else object):
    """状态栏组件"""
    
    def __init__(self, parent):
        if USE_CUSTOM_TK:
            super().__init__(parent, height=30, corner_radius=0)
        self.parent = parent
        
        # 状态标签
        self.labels = {}
        self._create_labels()
    
    def _create_labels(self):
        """创建状态标签"""
        if not USE_CUSTOM_TK:
            return
            
        items = [
            ("gpu", "🖥️ GPU: --"),
            ("speed", "⚡ 速度: --"),
            ("remaining", "⏱️ 剩余: --"),
            ("files", "📁 文件: 0/0"),
        ]
        
        for i, (key, text) in enumerate(items):
            label = ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=11))
            label.pack(side="left", padx=15, pady=5)
            self.labels[key] = label
    
    def update_status(self, gpu=None, speed=None, remaining=None, files=None):
        """更新状态"""
        if not USE_CUSTOM_TK:
            return
            
        if gpu is not None:
            self.labels["gpu"].configure(text=f"🖥️ GPU: {gpu}")
        if speed is not None:
            self.labels["speed"].configure(text=f"⚡ 速度: {speed}")
        if remaining is not None:
            self.labels["remaining"].configure(text=f"⏱️ 剩余: {remaining}")
        if files is not None:
            self.labels["files"].configure(text=f"📁 文件: {files}")


class FolderCard(ctk.CTkFrame if USE_CUSTOM_TK else object):
    """文件夹选择卡片"""
    
    def __init__(self, parent, label_text, icon, on_change=None):
        if USE_CUSTOM_TK:
            super().__init__(parent, fg_color=("gray90", "gray17"))
        
        self.on_change = on_change
        self.folder_path = ""
        
        if USE_CUSTOM_TK:
            self._create_ui(label_text, icon)
    
    def _create_ui(self, label_text, icon):
        """创建UI"""
        # 标题行
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        label = ctk.CTkLabel(
            header,
            text=f"{icon} {label_text}",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        label.pack(side="left")
        
        # 输入行
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.entry = ctk.CTkEntry(input_frame, height=35, placeholder_text="拖放文件夹或点击浏览...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.browse_btn = ctk.CTkButton(
            input_frame,
            text="📂 浏览",
            width=80,
            height=35,
            command=self._browse
        )
        self.browse_btn.pack(side="right")
        
        # 拖拽支持
        if USE_DND:
            self.entry.drop_target_register(DND_FILES)
            self.entry.dnd_bind('<<Drop>>', self._on_drop)
    
    def _browse(self):
        """浏览文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            self.set_path(folder)
    
    def _on_drop(self, event):
        """拖拽处理"""
        path = event.data.strip('{}')
        if os.path.isdir(path):
            self.set_path(path)
    
    def set_path(self, path):
        """设置路径"""
        self.folder_path = path
        if USE_CUSTOM_TK:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
        if self.on_change:
            self.on_change(path)
    
    def get_path(self):
        """获取路径"""
        if USE_CUSTOM_TK:
            return self.entry.get()
        return self.folder_path


class ModernOCRApp:
    """现代化OCR应用界面 v2.0"""
    
    def __init__(self):
        # 创建主窗口
        if USE_DND:
            self.root = TkinterDnD.Tk()
            ctk.set_appearance_mode("dark")
        elif USE_CUSTOM_TK:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        
        self.root.title("📄 PDF OCR 智能处理工具 v2.0")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        
        # 状态变量
        self.is_running = False
        self.msg_queue = queue.Queue()
        self.start_time = None
        
        # 当前设置
        self.settings = {
            'engine': 'hybrid',  # hybrid, paddle, deepseek
            'dpi': 150,
            'confidence_threshold': 0.85
        }
        
        # 创建界面
        if USE_CUSTOM_TK:
            self._create_modern_ui()
        else:
            self._create_classic_ui()
        
        # 定时检查消息队列
        self.root.after(100, self._check_queue)
    
    def _create_modern_ui(self):
        """创建现代化界面"""
        # 配置主窗口grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # ============ 侧边栏 ============
        self.sidebar = ctk.CTkFrame(self.root, width=180, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        self._create_sidebar()
        
        # ============ 主内容区 ============
        self.main_area = ctk.CTkFrame(self.root)
        self.main_area.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        
        # TabView
        self.tabview = ctk.CTkTabview(self.main_area, height=600)
        self.tabview.pack(fill="both", expand=True)
        
        self.tabview.add("📋 任务")
        self.tabview.add("📜 日志")
        self.tabview.add("📊 统计")
        
        self._create_task_tab()
        self._create_log_tab()
        self._create_stats_tab()
        
        # ============ 状态栏 ============
        self.statusbar = StatusBar(self.root)
        self.statusbar.grid(row=1, column=1, sticky="ew", padx=15, pady=(0, 10))
    
    def _create_sidebar(self):
        """创建侧边栏"""
        # Logo/标题
        logo_label = ctk.CTkLabel(
            self.sidebar,
            text="📄 PDF OCR",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        logo_label.pack(pady=(20, 5))
        
        version_label = ctk.CTkLabel(
            self.sidebar,
            text="v2.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        version_label.pack(pady=(0, 20))
        
        # 分隔线
        sep = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray40")
        sep.pack(fill="x", padx=20, pady=10)
        
        # OCR引擎选择
        engine_label = ctk.CTkLabel(self.sidebar, text="⚙️ OCR引擎", font=ctk.CTkFont(size=12))
        engine_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.engine_var = ctk.StringVar(value="hybrid")
        self.engine_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["hybrid", "paddle", "deepseek"],
            variable=self.engine_var,
            width=140,
            command=self._on_engine_change
        )
        self.engine_menu.pack(padx=20, pady=(0, 10))
        
        # DPI设置
        dpi_label = ctk.CTkLabel(self.sidebar, text="📐 DPI", font=ctk.CTkFont(size=12))
        dpi_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.dpi_var = ctk.StringVar(value="150")
        self.dpi_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["100", "150", "200", "300"],
            variable=self.dpi_var,
            width=140
        )
        self.dpi_menu.pack(padx=20, pady=(0, 10))
        
        # 置信度阈值
        conf_label = ctk.CTkLabel(self.sidebar, text="🎯 置信度阈值", font=ctk.CTkFont(size=12))
        conf_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.conf_slider = ctk.CTkSlider(
            self.sidebar,
            from_=0.5,
            to=1.0,
            number_of_steps=10,
            width=140
        )
        self.conf_slider.set(0.85)
        self.conf_slider.pack(padx=20, pady=(0, 5))
        
        self.conf_value_label = ctk.CTkLabel(self.sidebar, text="0.85", font=ctk.CTkFont(size=11))
        self.conf_value_label.pack(pady=(0, 10))
        self.conf_slider.configure(command=self._on_conf_change)
        
        # 分隔线
        sep2 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray40")
        sep2.pack(fill="x", padx=20, pady=10)
        
        # 主题切换
        theme_label = ctk.CTkLabel(self.sidebar, text="🎨 主题", font=ctk.CTkFont(size=12))
        theme_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.theme_switch = ctk.CTkSwitch(
            self.sidebar,
            text="暗色模式",
            command=self._toggle_theme
        )
        self.theme_switch.select()  # 默认暗色
        self.theme_switch.pack(padx=20, pady=(0, 20))
        
        # 底部空白填充
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)
        
        # 关于按钮
        about_btn = ctk.CTkButton(
            self.sidebar,
            text="ℹ️ 关于",
            fg_color="transparent",
            hover_color=("gray80", "gray30"),
            command=self._show_about
        )
        about_btn.pack(pady=(0, 20))
    
    def _create_task_tab(self):
        """创建任务选项卡"""
        tab = self.tabview.tab("📋 任务")
        
        # 文件夹选择区域
        folders_frame = ctk.CTkFrame(tab, fg_color="transparent")
        folders_frame.pack(fill="x", padx=10, pady=10)
        
        self.voucher_card = FolderCard(folders_frame, "凭证文件夹", "📁")
        self.voucher_card.pack(fill="x", pady=5)
        
        self.reference_card = FolderCard(folders_frame, "参照资料文件夹", "📂")
        self.reference_card.pack(fill="x", pady=5)
        
        self.output_card = FolderCard(folders_frame, "输出文件夹", "📤")
        self.output_card.pack(fill="x", pady=5)
        
        # 进度区域
        progress_frame = ctk.CTkFrame(tab)
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        progress_label = ctk.CTkLabel(
            progress_frame,
            text="处理进度",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        progress_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.current_file_label = ctk.CTkLabel(
            progress_frame,
            text="等待开始...",
            font=ctk.CTkFont(size=12)
        )
        self.current_file_label.pack(anchor="w", padx=15)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=500)
        self.progress_bar.pack(fill="x", padx=15, pady=10)
        self.progress_bar.set(0)
        
        self.progress_text = ctk.CTkLabel(
            progress_frame,
            text="0%",
            font=ctk.CTkFont(size=12)
        )
        self.progress_text.pack(pady=(0, 10))
        
        # 按钮区域
        button_frame = ctk.CTkFrame(tab, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="▶ 开始处理",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            width=200,
            command=self._start_processing
        )
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="⏹ 停止",
            font=ctk.CTkFont(size=16),
            height=50,
            width=100,
            fg_color="gray40",
            command=self._stop_processing,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        self.open_btn = ctk.CTkButton(
            button_frame,
            text="📁 打开输出",
            font=ctk.CTkFont(size=14),
            height=50,
            width=150,
            fg_color="green",
            command=self._open_output_folder
        )
        self.open_btn.pack(side="right", padx=5)
    
    def _create_log_tab(self):
        """创建日志选项卡"""
        tab = self.tabview.tab("📜 日志")
        
        # 日志工具栏
        toolbar = ctk.CTkFrame(tab, fg_color="transparent")
        toolbar.pack(fill="x", padx=10, pady=5)
        
        clear_btn = ctk.CTkButton(
            toolbar,
            text="🗑️ 清空",
            width=80,
            command=self._clear_log
        )
        clear_btn.pack(side="left", padx=5)
        
        export_btn = ctk.CTkButton(
            toolbar,
            text="💾 导出",
            width=80,
            command=self._export_log
        )
        export_btn.pack(side="left", padx=5)
        
        # 日志文本框
        self.log_textbox = ctk.CTkTextbox(tab, height=400)
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)
    
    def _create_stats_tab(self):
        """创建统计选项卡"""
        tab = self.tabview.tab("📊 统计")
        
        # 统计卡片
        stats_frame = ctk.CTkFrame(tab, fg_color="transparent")
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 行1: 文件统计
        row1 = ctk.CTkFrame(stats_frame, fg_color="transparent")
        row1.pack(fill="x", pady=5)
        
        self.stat_cards = {}
        
        stats_config = [
            ("total_files", "📁 总文件", "0"),
            ("processed", "✅ 已处理", "0"),
            ("pages", "📄 总页数", "0"),
            ("avg_time", "⏱️ 平均耗时", "-- s/页"),
        ]
        
        for key, title, value in stats_config:
            card = ctk.CTkFrame(row1, width=150, height=80)
            card.pack(side="left", fill="x", expand=True, padx=5)
            card.pack_propagate(False)
            
            title_label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12))
            title_label.pack(pady=(15, 5))
            
            value_label = ctk.CTkLabel(
                card,
                text=value,
                font=ctk.CTkFont(size=20, weight="bold")
            )
            value_label.pack()
            
            self.stat_cards[key] = value_label
        
        # 引擎使用统计
        engine_frame = ctk.CTkFrame(stats_frame)
        engine_frame.pack(fill="x", pady=20)
        
        engine_title = ctk.CTkLabel(
            engine_frame,
            text="🔧 引擎使用统计",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        engine_title.pack(anchor="w", padx=15, pady=10)
        
        self.paddle_bar = ctk.CTkProgressBar(engine_frame, width=400)
        self.paddle_bar.pack(fill="x", padx=15, pady=5)
        self.paddle_bar.set(0)
        
        self.paddle_label = ctk.CTkLabel(engine_frame, text="PaddleOCR: 0次 (0%)")
        self.paddle_label.pack(anchor="w", padx=15)
        
        self.deepseek_bar = ctk.CTkProgressBar(engine_frame, width=400)
        self.deepseek_bar.pack(fill="x", padx=15, pady=5)
        self.deepseek_bar.set(0)
        
        self.deepseek_label = ctk.CTkLabel(engine_frame, text="DeepSeek: 0次 (0%)")
        self.deepseek_label.pack(anchor="w", padx=15, pady=(0, 15))
    
    def _create_classic_ui(self):
        """创建经典界面(fallback)"""
        # 简化版本
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="PDF OCR 处理工具", font=("Arial", 18, "bold")).pack(pady=20)
        
        ttk.Label(frame, text="请安装 customtkinter 以获得完整界面:").pack()
        ttk.Label(frame, text="pip install customtkinter").pack(pady=10)
    
    # ============ 事件处理 ============
    
    def _on_engine_change(self, value):
        """引擎切换"""
        self.settings['engine'] = value
        self._log(f"OCR引擎切换为: {value}")
    
    def _on_conf_change(self, value):
        """置信度阈值变化"""
        self.settings['confidence_threshold'] = value
        self.conf_value_label.configure(text=f"{value:.2f}")
    
    def _toggle_theme(self):
        """切换主题"""
        if USE_CUSTOM_TK:
            current = ctk.get_appearance_mode()
            if current == "Dark":
                ctk.set_appearance_mode("light")
            else:
                ctk.set_appearance_mode("dark")
    
    def _show_about(self):
        """显示关于对话框"""
        messagebox.showinfo(
            "关于",
            "PDF OCR 智能处理工具 v2.0\n\n"
            "功能:\n"
            "• 混合OCR引擎 (Paddle + DeepSeek)\n"
            "• 智能置信度切换\n"
            "• 批量PDF处理\n"
            "• 文档分类匹配\n\n"
            "© 2026"
        )
    
    def _clear_log(self):
        """清空日志"""
        if USE_CUSTOM_TK:
            self.log_textbox.delete("1.0", "end")
    
    def _export_log(self):
        """导出日志"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt")]
        )
        if filepath:
            content = self.log_textbox.get("1.0", "end")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("成功", f"日志已导出到:\n{filepath}")
    
    def _log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        
        if USE_CUSTOM_TK:
            self.log_textbox.insert("end", log_msg)
            self.log_textbox.see("end")
    
    def _check_queue(self):
        """检查消息队列"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        
        self.root.after(100, self._check_queue)
    
    def _handle_message(self, msg):
        """处理消息"""
        msg_type = msg.get('type')
        
        if msg_type == 'log':
            self._log(msg['text'])
        elif msg_type == 'progress':
            value = msg['value'] / 100
            self.progress_bar.set(value)
            self.progress_text.configure(text=f"{msg['value']:.1f}%")
        elif msg_type == 'file':
            self.current_file_label.configure(text=f"当前: {msg['text']}")
        elif msg_type == 'status':
            self.statusbar.update_status(**msg.get('data', {}))
        elif msg_type == 'stats':
            self._update_stats(msg.get('data', {}))
        elif msg_type == 'done':
            self._processing_done(msg.get('success', True), msg.get('stats'))
    
    def _update_stats(self, data):
        """更新统计"""
        if 'total_files' in data:
            self.stat_cards['total_files'].configure(text=str(data['total_files']))
        if 'processed' in data:
            self.stat_cards['processed'].configure(text=str(data['processed']))
        if 'pages' in data:
            self.stat_cards['pages'].configure(text=str(data['pages']))
        if 'avg_time' in data:
            self.stat_cards['avg_time'].configure(text=f"{data['avg_time']:.1f}s/页")
    
    def _start_processing(self):
        """开始处理"""
        # 获取路径
        voucher = self.voucher_card.get_path()
        reference = self.reference_card.get_path()
        output = self.output_card.get_path()
        
        # 验证
        if not voucher or not os.path.isdir(voucher):
            messagebox.showerror("错误", "请选择有效的凭证文件夹")
            return
        
        if not reference or not os.path.isdir(reference):
            messagebox.showerror("错误", "请选择有效的参照资料文件夹")
            return
        
        # 默认输出目录
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(Path.home() / "Desktop" / f"OCR_结果_{timestamp}")
            self.output_card.set_path(output)
        
        # 更新UI
        self.is_running = True
        self.start_time = time.time()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)
        
        self._log("=" * 50)
        self._log(f"开始处理...")
        self._log(f"引擎: {self.settings['engine']}")
        self._log(f"DPI: {self.dpi_var.get()}")
        self._log(f"置信度阈值: {self.settings['confidence_threshold']:.2f}")
        
        # 启动处理线程
        thread = threading.Thread(
            target=self._run_processing,
            args=(voucher, reference, output),
            daemon=True
        )
        thread.start()
    
    def _run_processing(self, voucher, reference, output):
        """后台处理线程"""
        try:
            from run_ocr import run_ocr_pipeline_with_callback
            
            def callback(msg_type, **kwargs):
                if not self.is_running:
                    raise InterruptedError("用户取消")
                self.msg_queue.put({'type': msg_type, **kwargs})
            
            stats = run_ocr_pipeline_with_callback(
                voucher, reference, output,
                callback,
                engine=self.settings['engine'],
                dpi=int(self.dpi_var.get()),
                confidence_threshold=self.settings['confidence_threshold']
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
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if success and stats:
            self.progress_bar.set(1)
            self.progress_text.configure(text="100%")
            self.current_file_label.configure(text="✅ 处理完成!")
            
            self._log("=" * 50)
            self._log(f"处理完成! 总耗时: {elapsed/60:.1f}分钟")
            self._log(f"文件: {stats.get('total_files', 0)} | 页数: {stats.get('total_pages', 0)}")
            
            messagebox.showinfo("完成", 
                f"处理完成!\n\n"
                f"耗时: {elapsed/60:.1f}分钟\n"
                f"文件: {stats.get('total_files', 0)}\n"
                f"页数: {stats.get('total_pages', 0)}"
            )
        else:
            self.current_file_label.configure(text="已停止")
    
    def _open_output_folder(self):
        """打开输出文件夹"""
        folder = self.output_card.get_path()
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
