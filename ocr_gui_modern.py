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
    """中国传统配色：明快清爽"""
    # 背景色
    BG_MAIN = "#FDFBF7"          # 主背景 - 米宣纸
    BG_SIDEBAR = "#F8F9FA"       # 侧边栏 - 霜色 (极淡灰)
    BG_CARD = "#FFFFFF"          # 卡片 - 纯白
    
    # 主色调
    PRIMARY = "#F29C9F"          # 淡绯 (Danfei) - 柔和的粉红
    PRIMARY_HOVER = "#E08588"    # 淡绯加深

    # 强调色
    SUCCESS = "#78A355"          # 柳染 (Willow Green)
    SUCCESS_HOVER = "#5E8A3D"    # 柳染加深
    DANGER = "#D93A49"           # 赤红 (Crimson)
    SECONDARY = "#9D9D9D"        # 银鼠 (Silver Gray)

    # 文字
    TEXT_DARK = "#333333"        # 漆黑 - 正文
    TEXT_SIDEBAR = "#4A4A4A"     # 侧边栏深色文字
    TEXT_MUTED = "#888888"       # 浅灰 - 次要文字
    TEXT_ON_PRIMARY = "#2C2C2C"  # 按钮上的每字
    TEXT_ACCENT = "#D93A49"      # 强调文字

    # 边框
    BORDER = "#EFEFEF"

    # 圆角
    RADIUS = 10
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
            fg_color="#FAF9F6",
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
            hover_color="#7E7E7E",
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
        """创建现代化界面 - 顶部导航 + 双栏布局"""
        # 主布局Grid
        self.root.grid_columnconfigure(0, weight=4)  # 左侧操作区 (40%)
        self.root.grid_columnconfigure(1, weight=6)  # 右侧反馈区 (60%)
        self.root.grid_rowconfigure(0, weight=0)     # 顶部Header (固定高度)
        self.root.grid_rowconfigure(1, weight=1)     # 主体内容 (自适应)

        # ======== 顶部 Header ========
        self.header = ctk.CTkFrame(
            self.root, height=80, corner_radius=0,
            fg_color=Theme.BG_SIDEBAR # 使用霜色作为Header背景
        )
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._create_header()

        # ======== 左侧操作面板 ========
        self.left_panel = ctk.CTkFrame(self.root, fg_color=Theme.BG_MAIN, corner_radius=0)
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=30, pady=30)
        self._create_left_panel()

        # ======== 右侧反馈面板 ========
        self.right_panel = ctk.CTkFrame(self.root, fg_color=Theme.BG_MAIN, corner_radius=0)
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 30), pady=30)
        self._create_right_panel()

        # ======== 状态栏 ========
        self.statusbar = StatusBar(self.root)
        self.statusbar.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _create_header(self):
        """创建顶部导航栏"""
        # Logo (左侧)
        logo_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        logo_frame.pack(side="left", padx=30, pady=15)

        ctk.CTkLabel(
            logo_frame,
            text="📄 PDF OCR Pro",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=Theme.TEXT_DARK
        ).pack(anchor="w")

        ctk.CTkLabel(
            logo_frame,
            text="v3.0 · 智能识别", # 副标题
            font=ctk.CTkFont(size=12),
            text_color=Theme.TEXT_MUTED
        ).pack(anchor="w")

        # 设置区域 (右侧)
        # 使用一个胶囊状容器包裹设置项
        settings_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        settings_frame.pack(side="right", padx=30)

        # 1. 引擎
        self._create_header_setting(settings_frame, "引擎", ["hybrid", "rapid", "deepseek"],
                                  "hybrid", self._on_engine_change, width=100)

        # 间隔
        ctk.CTkFrame(settings_frame, width=20, height=1, fg_color="transparent").pack(side="left")

        # 2. DPI
        self.dpi_var = ctk.StringVar(value="150") # 需要保存引用
        self._create_header_setting(settings_frame, "DPI", ["100", "150", "200"],
                                  "150", None, variable=self.dpi_var, width=80)

        # 间隔
        ctk.CTkFrame(settings_frame, width=20, height=1, fg_color="transparent").pack(side="left")

        # 3. 阈值 (Slider)
        slider_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        slider_frame.pack(side="left")
        ctk.CTkLabel(slider_frame, text="阈值", font=ctk.CTkFont(size=11, weight="bold"), text_color=Theme.TEXT_MUTED).pack(anchor="w")

        self.conf_slider = ctk.CTkSlider(
            slider_frame, from_=0.5, to=1.0, number_of_steps=10, width=100,
            progress_color=Theme.PRIMARY, button_color=Theme.PRIMARY, button_hover_color=Theme.PRIMARY_HOVER,
            command=self._on_conf_change
        )
        self.conf_slider.pack(side="left", pady=(5,0))
        self.conf_slider.set(0.85)

        self.conf_value_label = ctk.CTkLabel(slider_frame, text="0.85", font=ctk.CTkFont(size=12), text_color=Theme.TEXT_DARK, width=35)
        self.conf_value_label.pack(side="left", padx=(5,0), pady=(2,0))

        # 间隔
        ctk.CTkFrame(settings_frame, width=20, height=1, fg_color="transparent").pack(side="left")

        # 3. 关于按钮
        ctk.CTkButton(
            settings_frame,
            text="ℹ️",
            width=40,
            height=36,
            corner_radius=Theme.RADIUS_SM,
            fg_color="transparent",
            hover_color="#E2E8F0",
            text_color=Theme.TEXT_SIDEBAR,
            font=ctk.CTkFont(size=16),
            command=self._show_about
        ).pack(side="left")

    def _create_header_setting(self, parent, label_text, values, default, command, variable=None, width=100):
        """创建Header中的单个设置项"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(side="left")

        ctk.CTkLabel(
            frame, text=label_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Theme.TEXT_MUTED
        ).pack(anchor="w", padx=2)

        if variable is None:
            if label_text == "引擎":
                self.engine_var = ctk.StringVar(value=default)
                variable = self.engine_var
            else:
                variable = ctk.StringVar(value=default)

        menu = ctk.CTkOptionMenu(
            frame,
            values=values,
            variable=variable,
            width=width,
            height=32,
            corner_radius=Theme.RADIUS_SM,
            fg_color="#FFFFFF",
            button_color="#F2F4F8",
            button_hover_color="#E2E8F0",
            text_color=Theme.TEXT_DARK,
            dropdown_fg_color="#FFFFFF",
            dropdown_text_color="#333333",
            font=ctk.CTkFont(size=13),
            command=command
        )
        menu.pack()
        return menu

    def _create_left_panel(self):
        """左侧：输入与操作"""
        # 1. 文件输入区 (卡片堆叠)
        input_group = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        input_group.pack(fill="x", pady=(0, 20))

        self.voucher_card = FolderInputCard(input_group, "凭证文件夹", "📁")
        self.voucher_card.pack(fill="x", pady=(0, 15))

        self.reference_card = FolderInputCard(input_group, "参照资料文件夹", "📂")
        self.reference_card.pack(fill="x", pady=(0, 15))

        self.output_card = FolderInputCard(input_group, "输出文件夹", "📤")
        self.output_card.pack(fill="x", pady=(0, 15))

        # 2. 进度条 (醒目)
        progress_card = Card(self.left_panel)
        progress_card.pack(fill="x", pady=(0, 20))
        
        # 进度头
        p_head = ctk.CTkFrame(progress_card, fg_color="transparent")
        p_head.pack(fill="x", padx=20, pady=(15, 5))
        self.current_file_label = ctk.CTkLabel(p_head, text="准备就绪", text_color=Theme.TEXT_DARK, font=ctk.CTkFont(size=13))
        self.current_file_label.pack(side="left")
        self.progress_text = ctk.CTkLabel(p_head, text="0%", text_color=Theme.TEXT_ACCENT, font=ctk.CTkFont(size=14, weight="bold"))
        self.progress_text.pack(side="right") # 确保这个变量名在update时能找到

        self.progress_bar = ctk.CTkProgressBar(
            progress_card, height=16, corner_radius=8,
            progress_color=Theme.PRIMARY, fg_color="#F3F4F6"
        )
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 20))
        self.progress_bar.set(0)

        # 3. 核心操作按钮 (底部)
        action_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        action_frame.pack(fill="x", pady=(10, 0))

        self.start_btn = ctk.CTkButton(
            action_frame,
            text="▶ 开始处理",
            height=56,
            corner_radius=Theme.RADIUS,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=Theme.PRIMARY, hover_color=Theme.PRIMARY_HOVER,
            text_color=Theme.TEXT_ON_PRIMARY,
            command=self._start_processing
        )
        self.start_btn.pack(fill="x", pady=(0, 10))

        # 辅助按钮行
        sub_actioms = ctk.CTkFrame(action_frame, fg_color="transparent")
        sub_actioms.pack(fill="x")

        self.stop_btn = ctk.CTkButton(
            sub_actioms, text="⏹ 停止",
            fg_color="#FEE2E2", text_color=Theme.DANGER, hover_color="#FECACA", # 浅红背景
            width=100, height=40,
            command=self._stop_processing, state="disabled"
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))

        self.open_btn = ctk.CTkButton(
            sub_actioms, text="📂 打开输出",
            fg_color="transparent", border_width=1, border_color=Theme.BORDER,
            text_color=Theme.TEXT_SIDEBAR, hover_color="#F3F4F6",
            width=100, height=40,
            command=self._open_output_folder
        )
        self.open_btn.pack(side="left", expand=True, fill="x")

    def _create_right_panel(self):
        """右侧：反馈与日志"""
        # 使用Tabview来组织信息
        self.tab_view = ctk.CTkTabview(
            self.right_panel,
            corner_radius=Theme.RADIUS,
            fg_color=Theme.BG_CARD,
            segmented_button_fg_color="#F3F4F6",
            segmented_button_selected_color=Theme.PRIMARY,
            segmented_button_selected_hover_color=Theme.PRIMARY_HOVER,
            segmented_button_unselected_color="#F3F4F6",
            segmented_button_unselected_hover_color="#E5E7EB",
            text_color=Theme.TEXT_DARK
        )
        self.tab_view.pack(fill="both", expand=True)

        self.tab_view.add("📜 运行日志")
        self.tab_view.add("📊 统计数据")

        # === 日志 Tab ===
        # 工具栏
        log_tools = ctk.CTkFrame(self.tab_view.tab("📜 运行日志"), fg_color="transparent")
        log_tools.pack(fill="x", pady=(0, 10))

        ctk.CTkButton(log_tools, text="清空日志", height=28, width=80,
                     fg_color="transparent", border_width=1, border_color=Theme.BORDER, text_color=Theme.TEXT_MUTED,
                     command=self._clear_log).pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            self.tab_view.tab("📜 运行日志"),
            corner_radius=Theme.RADIUS_SM,
            fg_color="#F9FAFB",
            text_color=Theme.TEXT_DARK,
            font=ctk.CTkFont(family="Consolas", size=12),
            activate_scrollbars=True
        )
        self.log_textbox.pack(fill="both", expand=True)

        # === 统计 Tab ===
        stats_frame = self.tab_view.tab("📊 统计数据")

        # 统计卡片 Grid
        self.stat_cards = {}
        grid_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        grid_frame.pack(fill="x", pady=20)

        stats_config = [
            ("total_files", "📁 总文件", "0", Theme.TEXT_ACCENT),
            ("processed", "✅ 已处理", "0", Theme.SUCCESS),
            ("pages", "📄 总页数", "0", "#5B5EA6"),
            ("avg_time", "⏱️ 平均耗时", "--", "#E9BB1D"),
        ]
        
        for i, (key, title, val, col) in enumerate(stats_config):
            card = ctk.CTkFrame(grid_frame, fg_color="#F3F4F6", corner_radius=Theme.RADIUS)
            card.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="ew")
            grid_frame.columnconfigure(i%2, weight=1)

            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12), text_color=Theme.TEXT_MUTED).pack(pady=(15, 5))
            lbl = ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=24, weight="bold"), text_color=col)
            lbl.pack(pady=(0, 15))
            self.stat_cards[key] = lbl

        # 引擎统计
        eng_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        eng_frame.pack(fill="x", pady=20, padx=10)

        ctk.CTkLabel(eng_frame, text="引擎调用分布", font=ctk.CTkFont(size=14, weight="bold"), text_color=Theme.TEXT_DARK).pack(anchor="w", pady=(0,10))

        self.rapid_label = ctk.CTkLabel(eng_frame, text="RapidOCR: 0", text_color=Theme.TEXT_MUTED, anchor="w")
        self.rapid_label.pack(fill="x")
        self.rapid_bar = ctk.CTkProgressBar(eng_frame, height=8, progress_color="#3B82F6", fg_color="#E5E7EB")
        self.rapid_bar.pack(fill="x", pady=(5, 15))
        self.rapid_bar.set(0)
        
        self.deepseek_label = ctk.CTkLabel(eng_frame, text="DeepSeek: 0", text_color=Theme.TEXT_MUTED, anchor="w")
        self.deepseek_label.pack(fill="x")
        self.deepseek_bar = ctk.CTkProgressBar(eng_frame, height=8, progress_color="#8B5CF6", fg_color="#E5E7EB")
        self.deepseek_bar.pack(fill="x", pady=(5, 0))
        self.deepseek_bar.set(0)

# 保留辅助方法（事件处理等），并适配新布局
    def _create_sidebar(self): pass # 废弃
    def _create_sidebar_section(self, t): pass # 废弃
    def _create_main_content(self): pass # 废弃
    def _create_task_tab(self): pass # 废弃
    def _create_log_tab(self): pass # 废弃
    def _create_stats_tab(self): pass # 废弃
    def _on_tab_change(self, value): pass # 废弃，现在使用Tabview自动管理

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
