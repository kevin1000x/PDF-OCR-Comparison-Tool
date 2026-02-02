"""
PDF OCR处理工具 - 现代化GUI v3.0
=================================

精致设计版本 - 参照SaaS风格
- 深色侧边栏 + 浅灰主背景 + 白色卡片
- 优化配色、间距、层次感
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
    ctk.set_appearance_mode("light")  # 使用亮色以展示配色
    ctk.set_default_color_theme("blue")
    USE_CUSTOM_TK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk
    USE_CUSTOM_TK = False
    print("提示: pip install customtkinter")

from tkinter import filedialog, messagebox

# 尝试导入拖拽支持
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    USE_DND = True
except ImportError:
    USE_DND = False

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ 配色方案 ============
class Theme:
    """SaaS风格配色"""
    # 背景色
    BG_MAIN = "#F3F4F6"          # 主背景 - 浅灰蓝
    BG_SIDEBAR = "#1E293B"       # 侧边栏 - 深午夜蓝
    BG_CARD = "#FFFFFF"          # 卡片 - 纯白
    
    # 主色调
    PRIMARY = "#4F46E5"          # 靛青色
    PRIMARY_HOVER = "#4338CA"    # 靛青色悬停
    SUCCESS = "#10B981"          # 翡翠绿
    SUCCESS_HOVER = "#059669"
    DANGER = "#EF4444"           # 红色
    SECONDARY = "#6B7280"        # 次要灰
    
    # 文字
    TEXT_DARK = "#1F2937"        # 深色文字
    TEXT_LIGHT = "#FFFFFF"       # 浅色文字
    TEXT_MUTED = "#9CA3AF"       # 次要文字
    
    # 边框
    BORDER = "#E5E7EB"
    
    # 圆角
    RADIUS = 8
    RADIUS_SM = 6


# ============ 卡片组件 ============
class Card(ctk.CTkFrame if USE_CUSTOM_TK else object):
    """白色卡片容器"""
    def __init__(self, parent, **kwargs):
        if USE_CUSTOM_TK:
            super().__init__(
                parent,
                fg_color=Theme.BG_CARD,
                corner_radius=Theme.RADIUS,
                **kwargs
            )


class FolderInputCard(ctk.CTkFrame if USE_CUSTOM_TK else object):
    """文件夹输入卡片 - 虚线边框风格"""
    
    def __init__(self, parent, title, icon="📁", on_change=None):
        if USE_CUSTOM_TK:
            super().__init__(parent, fg_color=Theme.BG_CARD, corner_radius=Theme.RADIUS)
        
        self.on_change = on_change
        self.folder_path = ""
        
        if USE_CUSTOM_TK:
            self._create_ui(title, icon)
    
    def _create_ui(self, title, icon):
        # 标题
        title_label = ctk.CTkLabel(
            self,
            text=f"{icon}  {title}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=Theme.TEXT_DARK,
            anchor="w"
        )
        title_label.pack(fill="x", padx=16, pady=(16, 8))
        
        # 输入区域 - 整合按钮
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=16, pady=(0, 16))
        
        self.entry = ctk.CTkEntry(
            input_frame,
            height=40,
            corner_radius=Theme.RADIUS_SM,
            border_width=1,
            border_color=Theme.BORDER,
            fg_color="#F9FAFB",
            placeholder_text="拖放文件夹或点击右侧浏览...",
            placeholder_text_color=Theme.TEXT_MUTED
        )
        self.entry.pack(side="left", fill="x", expand=True)
        
        self.browse_btn = ctk.CTkButton(
            input_frame,
            text="浏览",
            width=70,
            height=40,
            corner_radius=Theme.RADIUS_SM,
            fg_color=Theme.SECONDARY,
            hover_color="#4B5563",
            command=self._browse
        )
        self.browse_btn.pack(side="right", padx=(8, 0))
        
        # 拖拽支持
        if USE_DND:
            self.entry.drop_target_register(DND_FILES)
            self.entry.dnd_bind('<<Drop>>', self._on_drop)
    
    def _browse(self):
        folder = filedialog.askdirectory()
        if folder:
            self.set_path(folder)
    
    def _on_drop(self, event):
        path = event.data.strip('{}')
        if os.path.isdir(path):
            self.set_path(path)
    
    def set_path(self, path):
        self.folder_path = path
        if USE_CUSTOM_TK:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
        if self.on_change:
            self.on_change(path)
    
    def get_path(self):
        if USE_CUSTOM_TK:
            return self.entry.get()
        return self.folder_path


# ============ 状态栏 ============
class StatusBar(ctk.CTkFrame if USE_CUSTOM_TK else object):
    """底部状态栏"""
    
    def __init__(self, parent):
        if USE_CUSTOM_TK:
            super().__init__(parent, height=36, fg_color=Theme.BG_CARD, corner_radius=0)
        
        self.labels = {}
        self._create_labels()
    
    def _create_labels(self):
        if not USE_CUSTOM_TK:
            return
            
        items = [
            ("gpu", "🖥️ GPU: --"),
            ("speed", "⚡ 速度: --"),
            ("remaining", "⏱️ 剩余: --"),
            ("files", "📁 文件: 0/0"),
        ]
        
        for key, text in items:
            label = ctk.CTkLabel(
                self, text=text,
                font=ctk.CTkFont(size=11),
                text_color=Theme.TEXT_MUTED
            )
            label.pack(side="left", padx=20, pady=8)
            self.labels[key] = label
    
    def update_status(self, **kwargs):
        if not USE_CUSTOM_TK:
            return
        for key, value in kwargs.items():
            if key in self.labels and value is not None:
                icons = {"gpu": "🖥️", "speed": "⚡", "remaining": "⏱️", "files": "📁"}
                self.labels[key].configure(text=f"{icons.get(key, '')} {key.title()}: {value}")


# ============ 主应用 ============
class ModernOCRApp:
    """现代化OCR应用界面 v3.0"""
    
    def __init__(self):
        if USE_DND:
            self.root = TkinterDnD.Tk()
        elif USE_CUSTOM_TK:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        
        self.root.title("PDF OCR Pro")
        self.root.geometry("1150x780")
        self.root.minsize(950, 650)
        
        if USE_CUSTOM_TK:
            self.root.configure(fg_color=Theme.BG_MAIN)
        
        # 状态变量
        self.is_running = False
        self.msg_queue = queue.Queue()
        self.start_time = None
        
        # 设置
        self.settings = {
            'engine': 'hybrid',
            'dpi': 150,
            'confidence_threshold': 0.85
        }
        
        if USE_CUSTOM_TK:
            self._create_modern_ui()
        
        self.root.after(100, self._check_queue)
    
    def _create_modern_ui(self):
        """创建现代化界面"""
        # 主布局
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # ======== 深色侧边栏 ========
        self.sidebar = ctk.CTkFrame(
            self.root, width=220, corner_radius=0,
            fg_color=Theme.BG_SIDEBAR
        )
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._create_sidebar()
        
        # ======== 主内容区 ========
        self.main_frame = ctk.CTkFrame(self.root, fg_color=Theme.BG_MAIN, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self._create_main_content()
        
        # ======== 状态栏 ========
        self.statusbar = StatusBar(self.root)
        self.statusbar.grid(row=1, column=1, sticky="ew")
    
    def _create_sidebar(self):
        """创建深色侧边栏"""
        # Logo区域
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(30, 10))
        
        logo_label = ctk.CTkLabel(
            logo_frame,
            text="📄 PDF OCR Pro",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=Theme.TEXT_LIGHT
        )
        logo_label.pack(anchor="w")
        
        version_label = ctk.CTkLabel(
            logo_frame,
            text="v3.0 · 智能识别",
            font=ctk.CTkFont(size=11),
            text_color=Theme.TEXT_MUTED
        )
        version_label.pack(anchor="w", pady=(2, 0))
        
        # 间距
        ctk.CTkFrame(self.sidebar, height=30, fg_color="transparent").pack()
        
        # OCR引擎
        self._create_sidebar_section("⚙️ OCR引擎")
        self.engine_var = ctk.StringVar(value="hybrid")
        self.engine_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["hybrid", "rapid", "deepseek"],
            variable=self.engine_var,
            width=180,
            height=36,
            corner_radius=Theme.RADIUS_SM,
            fg_color="#334155",
            button_color="#475569",
            button_hover_color="#64748B",
            dropdown_fg_color="#1E293B",
            command=self._on_engine_change
        )
        self.engine_menu.pack(padx=20, pady=(0, 20))
        
        # DPI设置
        self._create_sidebar_section("📐 DPI")
        self.dpi_var = ctk.StringVar(value="150")
        self.dpi_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["100", "150", "200", "300"],
            variable=self.dpi_var,
            width=180,
            height=36,
            corner_radius=Theme.RADIUS_SM,
            fg_color="#334155",
            button_color="#475569",
            button_hover_color="#64748B",
            dropdown_fg_color="#1E293B"
        )
        self.dpi_menu.pack(padx=20, pady=(0, 20))
        
        # 置信度阈值
        self._create_sidebar_section("🎯 置信度阈值")
        
        conf_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        conf_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.conf_slider = ctk.CTkSlider(
            conf_frame,
            from_=0.5,
            to=1.0,
            number_of_steps=10,
            width=140,
            progress_color=Theme.PRIMARY,
            button_color=Theme.PRIMARY,
            button_hover_color=Theme.PRIMARY_HOVER
        )
        self.conf_slider.set(0.85)
        self.conf_slider.pack(side="left")
        
        self.conf_value_label = ctk.CTkLabel(
            conf_frame,
            text="0.85",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=Theme.TEXT_LIGHT,
            width=40
        )
        self.conf_value_label.pack(side="right")
        self.conf_slider.configure(command=self._on_conf_change)
        
        # 底部填充
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)
        
        # 关于按钮
        about_btn = ctk.CTkButton(
            self.sidebar,
            text="ℹ️  关于",
            fg_color="transparent",
            hover_color="#334155",
            text_color=Theme.TEXT_MUTED,
            anchor="w",
            height=40,
            command=self._show_about
        )
        about_btn.pack(fill="x", padx=15, pady=(0, 30))
    
    def _create_sidebar_section(self, title):
        """创建侧边栏分区标题"""
        label = ctk.CTkLabel(
            self.sidebar,
            text=title,
            font=ctk.CTkFont(size=12),
            text_color=Theme.TEXT_MUTED,
            anchor="w"
        )
        label.pack(fill="x", padx=20, pady=(0, 8))
    
    def _create_main_content(self):
        """创建主内容区"""
        # 顶部标签导航 - 使用SegmentedButton
        nav_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        nav_frame.pack(fill="x", padx=30, pady=(25, 15))
        
        self.tab_var = ctk.StringVar(value="任务")
        self.tab_buttons = ctk.CTkSegmentedButton(
            nav_frame,
            values=["📋 任务", "📜 日志", "📊 统计"],
            variable=self.tab_var,
            font=ctk.CTkFont(size=13),
            fg_color=Theme.BG_CARD,
            selected_color=Theme.PRIMARY,
            selected_hover_color=Theme.PRIMARY_HOVER,
            unselected_color=Theme.BG_CARD,
            unselected_hover_color="#E5E7EB",
            corner_radius=Theme.RADIUS,
            command=self._on_tab_change
        )
        self.tab_buttons.pack(side="left")
        
        # 内容容器
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        
        # 创建各Tab内容
        self.tab_frames = {}
        self._create_task_tab()
        self._create_log_tab()
        self._create_stats_tab()
        
        # 默认显示任务Tab
        self._show_tab("📋 任务")
    
    def _create_task_tab(self):
        """任务选项卡"""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.tab_frames["📋 任务"] = frame
        
        # 文件夹输入卡片
        self.voucher_card = FolderInputCard(frame, "凭证文件夹", "📁")
        self.voucher_card.pack(fill="x", pady=(0, 12))
        
        self.reference_card = FolderInputCard(frame, "参照资料文件夹", "📂")
        self.reference_card.pack(fill="x", pady=(0, 12))
        
        self.output_card = FolderInputCard(frame, "输出文件夹", "📤")
        self.output_card.pack(fill="x", pady=(0, 20))
        
        # 进度卡片
        progress_card = Card(frame)
        progress_card.pack(fill="x", pady=(0, 20))
        
        # 进度标题行
        progress_header = ctk.CTkFrame(progress_card, fg_color="transparent")
        progress_header.pack(fill="x", padx=16, pady=(16, 8))
        
        self.current_file_label = ctk.CTkLabel(
            progress_header,
            text="等待开始...",
            font=ctk.CTkFont(size=13),
            text_color=Theme.TEXT_DARK,
            anchor="w"
        )
        self.current_file_label.pack(side="left")
        
        self.progress_text = ctk.CTkLabel(
            progress_header,
            text="0%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=Theme.PRIMARY
        )
        self.progress_text.pack(side="right")
        
        # 进度条
        self.progress_bar = ctk.CTkProgressBar(
            progress_card,
            height=14,
            corner_radius=7,
            progress_color=Theme.PRIMARY,
            fg_color="#E5E7EB"
        )
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 16))
        self.progress_bar.set(0)
        
        # 按钮区域
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(fill="x")
        
        # 开始按钮 - 主要操作
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="▶  开始处理",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=52,
            width=220,
            corner_radius=Theme.RADIUS,
            fg_color=Theme.PRIMARY,
            hover_color=Theme.PRIMARY_HOVER,
            command=self._start_processing
        )
        self.start_btn.pack(side="left")
        
        # 停止按钮 - Ghost风格
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="⏹  停止",
            font=ctk.CTkFont(size=14),
            height=52,
            width=100,
            corner_radius=Theme.RADIUS,
            fg_color="transparent",
            border_width=1,
            border_color=Theme.SECONDARY,
            text_color=Theme.SECONDARY,
            hover_color="#F3F4F6",
            command=self._stop_processing,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(12, 0))
        
        # 打开输出 - 成功色，初始弱化
        self.open_btn = ctk.CTkButton(
            button_frame,
            text="📂  打开输出",
            font=ctk.CTkFont(size=14),
            height=52,
            width=140,
            corner_radius=Theme.RADIUS,
            fg_color=Theme.SECONDARY,
            hover_color="#4B5563",
            command=self._open_output_folder
        )
        self.open_btn.pack(side="right")
    
    def _create_log_tab(self):
        """日志选项卡"""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.tab_frames["📜 日志"] = frame
        
        # 日志卡片
        log_card = Card(frame)
        log_card.pack(fill="both", expand=True)
        
        # 工具栏
        toolbar = ctk.CTkFrame(log_card, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(16, 8))
        
        ctk.CTkButton(
            toolbar, text="🗑️ 清空", width=80, height=32,
            fg_color=Theme.SECONDARY, hover_color="#4B5563",
            command=self._clear_log
        ).pack(side="left", padx=(0, 8))
        
        ctk.CTkButton(
            toolbar, text="💾 导出", width=80, height=32,
            fg_color=Theme.SECONDARY, hover_color="#4B5563",
            command=self._export_log
        ).pack(side="left")
        
        # 日志文本框
        self.log_textbox = ctk.CTkTextbox(
            log_card, height=400,
            corner_radius=Theme.RADIUS_SM,
            fg_color="#F9FAFB",
            text_color=Theme.TEXT_DARK,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.log_textbox.pack(fill="both", expand=True, padx=16, pady=(0, 16))
    
    def _create_stats_tab(self):
        """统计选项卡"""
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.tab_frames["📊 统计"] = frame
        
        # 统计卡片行
        stats_row = ctk.CTkFrame(frame, fg_color="transparent")
        stats_row.pack(fill="x", pady=(0, 20))
        
        self.stat_cards = {}
        stats_config = [
            ("total_files", "📁 总文件", "0", Theme.PRIMARY),
            ("processed", "✅ 已处理", "0", Theme.SUCCESS),
            ("pages", "📄 总页数", "0", "#8B5CF6"),
            ("avg_time", "⏱️ 平均耗时", "-- s", "#F59E0B"),
        ]
        
        for key, title, value, color in stats_config:
            card = Card(stats_row)
            card.pack(side="left", fill="x", expand=True, padx=(0, 12) if key != "avg_time" else 0)
            
            ctk.CTkLabel(
                card, text=title,
                font=ctk.CTkFont(size=12),
                text_color=Theme.TEXT_MUTED
            ).pack(pady=(20, 5))
            
            value_label = ctk.CTkLabel(
                card, text=value,
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color=color
            )
            value_label.pack(pady=(0, 20))
            
            self.stat_cards[key] = value_label
        
        # 引擎统计卡片
        engine_card = Card(frame)
        engine_card.pack(fill="x")
        
        ctk.CTkLabel(
            engine_card,
            text="🔧 引擎使用统计",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=Theme.TEXT_DARK,
            anchor="w"
        ).pack(fill="x", padx=20, pady=(20, 15))
        
        # RapidOCR进度
        self.rapid_label = ctk.CTkLabel(
            engine_card, text="RapidOCR: 0次 (0%)",
            font=ctk.CTkFont(size=12),
            text_color=Theme.TEXT_MUTED, anchor="w"
        )
        self.rapid_label.pack(fill="x", padx=20)
        
        self.rapid_bar = ctk.CTkProgressBar(
            engine_card, height=10, corner_radius=5,
            progress_color="#3B82F6", fg_color="#E5E7EB"
        )
        self.rapid_bar.pack(fill="x", padx=20, pady=(5, 15))
        self.rapid_bar.set(0)
        
        # DeepSeek进度
        self.deepseek_label = ctk.CTkLabel(
            engine_card, text="DeepSeek: 0次 (0%)",
            font=ctk.CTkFont(size=12),
            text_color=Theme.TEXT_MUTED, anchor="w"
        )
        self.deepseek_label.pack(fill="x", padx=20)
        
        self.deepseek_bar = ctk.CTkProgressBar(
            engine_card, height=10, corner_radius=5,
            progress_color="#8B5CF6", fg_color="#E5E7EB"
        )
        self.deepseek_bar.pack(fill="x", padx=20, pady=(5, 20))
        self.deepseek_bar.set(0)
    
    def _on_tab_change(self, value):
        """切换Tab"""
        self._show_tab(value)
    
    def _show_tab(self, tab_name):
        """显示指定Tab"""
        for name, frame in self.tab_frames.items():
            if name == tab_name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
    
    # ============ 事件处理 ============
    
    def _on_engine_change(self, value):
        self.settings['engine'] = value
        self._log(f"OCR引擎切换为: {value}")
    
    def _on_conf_change(self, value):
        self.settings['confidence_threshold'] = value
        self.conf_value_label.configure(text=f"{value:.2f}")
    
    def _show_about(self):
        messagebox.showinfo(
            "关于 PDF OCR Pro",
            "📄 PDF OCR Pro v3.0\n\n"
            "智能文档识别系统\n\n"
            "• 混合OCR引擎 (Paddle + DeepSeek)\n"
            "• 智能置信度切换\n"
            "• 批量PDF处理\n"
            "• 文档分类匹配\n\n"
            "© 2026"
        )
    
    def _clear_log(self):
        if USE_CUSTOM_TK:
            self.log_textbox.delete("1.0", "end")
    
    def _export_log(self):
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
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        if USE_CUSTOM_TK:
            self.log_textbox.insert("end", log_msg)
            self.log_textbox.see("end")
    
    def _check_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._check_queue)
    
    def _handle_message(self, msg):
        msg_type = msg.get('type')
        
        if msg_type == 'log':
            self._log(msg['text'])
        elif msg_type == 'progress':
            value = msg['value'] / 100
            self.progress_bar.set(value)
            self.progress_text.configure(text=f"{msg['value']:.1f}%")
        elif msg_type == 'file':
            self.current_file_label.configure(text=f"处理: {msg['text']}")
        elif msg_type == 'done':
            self._processing_done(msg.get('success', True), msg.get('stats'))
    
    def _start_processing(self):
        voucher = self.voucher_card.get_path()
        reference = self.reference_card.get_path()
        output = self.output_card.get_path()
        
        if not voucher or not os.path.isdir(voucher):
            messagebox.showerror("错误", "请选择有效的凭证文件夹")
            return
        
        if not reference or not os.path.isdir(reference):
            messagebox.showerror("错误", "请选择有效的参照资料文件夹")
            return
        
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = str(Path.home() / "Desktop" / f"OCR_结果_{timestamp}")
            self.output_card.set_path(output)
        
        # 更新UI状态
        self.is_running = True
        self.start_time = time.time()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)
        
        # 打开输出按钮保持弱化
        self.open_btn.configure(fg_color=Theme.SECONDARY)
        
        self._log("=" * 50)
        self._log(f"开始处理...")
        self._log(f"引擎: {self.settings['engine']} | DPI: {self.dpi_var.get()}")
        
        # 启动处理线程
        thread = threading.Thread(
            target=self._run_processing,
            args=(voucher, reference, output),
            daemon=True
        )
        thread.start()
    
    def _run_processing(self, voucher, reference, output):
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
        self.is_running = False
        self._log("正在停止...")
    
    def _processing_done(self, success, stats=None):
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if success and stats:
            self.progress_bar.set(1)
            self.progress_text.configure(text="100%")
            self.current_file_label.configure(text="✅ 处理完成!")
            
            # 打开输出按钮高亮
            self.open_btn.configure(fg_color=Theme.SUCCESS, hover_color=Theme.SUCCESS_HOVER)
            
            self._log("=" * 50)
            self._log(f"处理完成! 总耗时: {elapsed/60:.1f}分钟")
            
            # 更新统计Tab
            self._update_stats_tab(stats, elapsed)
            
            # 显示详细统计
            total_files = stats.get('voucher_files', 0) + stats.get('reference_files', 0)
            matched = stats.get('matched', 0)
            partial = stats.get('partial', 0)
            unmatched = stats.get('unmatched', 0)
            
            messagebox.showinfo("完成", 
                f"处理完成!\n\n"
                f"耗时: {elapsed/60:.1f}分钟\n"
                f"凭证文件: {stats.get('voucher_files', 0)}\n"
                f"参照文件: {stats.get('reference_files', 0)}\n\n"
                f"匹配结果:\n"
                f"  ✅ 匹配: {matched}\n"
                f"  🔶 部分匹配: {partial}\n"
                f"  ❌ 未匹配: {unmatched}"
            )
        else:
            self.current_file_label.configure(text="已停止")
    
    def _update_stats_tab(self, stats, elapsed):
        """更新统计Tab的数据"""
        if not USE_CUSTOM_TK:
            return
        
        total_files = stats.get('voucher_files', 0) + stats.get('reference_files', 0)
        total_pages = stats.get('voucher_pages', 0) + stats.get('reference_pages', 0)
        matched = stats.get('matched', 0)
        
        # 更新统计卡片
        if hasattr(self, 'stat_cards'):
            if 'total_files' in self.stat_cards:
                self.stat_cards['total_files'].configure(text=str(total_files))
            if 'processed' in self.stat_cards:
                self.stat_cards['processed'].configure(text=str(matched))
            if 'pages' in self.stat_cards:
                self.stat_cards['pages'].configure(text=str(total_pages))
            if 'avg_time' in self.stat_cards:
                avg = elapsed / total_files if total_files > 0 else 0
                self.stat_cards['avg_time'].configure(text=f"{avg:.1f}s")
        
        # 更新引擎统计（如果有混合引擎统计数据）
        rapid_calls = stats.get('rapid_calls', 0)
        deepseek_calls = stats.get('deepseek_calls', 0)
        total_ocr = rapid_calls + deepseek_calls
        
        if total_ocr > 0 and hasattr(self, 'rapid_label'):
            rapid_pct = rapid_calls / total_ocr
            self.rapid_label.configure(text=f"RapidOCR: {rapid_calls}次 ({rapid_pct:.0%})")
            self.rapid_bar.set(rapid_pct)
            
            deepseek_pct = deepseek_calls / total_ocr
            self.deepseek_label.configure(text=f"DeepSeek: {deepseek_calls}次 ({deepseek_pct:.0%})")
            self.deepseek_bar.set(deepseek_pct)
    
    def _open_output_folder(self):
        folder = self.output_card.get_path()
        if folder and os.path.isdir(folder):
            os.startfile(folder)
        else:
            messagebox.showwarning("提示", "输出文件夹不存在")
    
    def run(self):
        self.root.mainloop()


def main():
    app = ModernOCRApp()
    app.run()


if __name__ == "__main__":
    main()
