# PDF-OCR-Comparison-Tool

智能PDF OCR识别与文档比对工具 - 使用DeepSeek-OCR2进行高精度中文文档识别

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 功能特点

- 🔍 **高精度OCR识别** - 基于DeepSeek-OCR2模型，支持复杂中文文档
- 📄 **可搜索PDF生成** - 将扫描PDF转换为可搜索的PDF（嵌入文本层）
- 🔗 **智能文档比对** - 自动匹配凭证与参照资料，生成对比报告
- 🖥️ **图形化界面** - 简单易用的桌面GUI，支持实时进度显示
- ⚡ **GPU加速** - 支持CUDA加速，大幅提升处理速度
- 💾 **缓存机制** - 避免重复处理，支持断点续传

## 🚀 快速开始

### 环境要求

- Windows 10/11
- Python 3.9+
- NVIDIA GPU（推荐RTX 3060以上，12GB显存）
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)（PDF转图像）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/PDF-OCR-Comparison-Tool.git
cd PDF-OCR-Comparison-Tool

# 2. 创建虚拟环境
conda create -n pdf-ocr python=3.10
conda activate pdf-ocr

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装Poppler（Windows）
# 下载后解压到 C:\poppler\
```

### 使用方法

#### 方式一：图形界面（推荐）

```bash
python ocr_gui.py
```

或双击 `启动OCR工具.bat`

![GUI界面](docs/gui_screenshot.png)

#### 方式二：命令行

```bash
# 基本用法
python run_ocr.py "凭证文件夹" "参照资料文件夹"

# 指定输出目录
python run_ocr.py "凭证文件夹" "参照资料文件夹" "输出目录"
```

#### 方式三：完整功能（高级）

```bash
# 构建索引
python main.py --index-only

# 完整处理
python main.py

# 指定项目
python main.py --project "项目名称"
```

## 📁 项目结构

```
PDF-OCR-Comparison-Tool/
├── 核心模块
│   ├── deepseek_ocr2_engine.py  # DeepSeek-OCR2引擎
│   ├── ocr_engine.py            # PaddleOCR引擎（备选）
│   ├── pdf_processor.py         # PDF处理模块
│   ├── document_classifier.py   # 文档分类器
│   ├── content_matcher.py       # 内容比对模块
│   ├── project_detector.py      # 项目检测器
│   └── ocr_cache.py             # OCR结果缓存
│
├── 应用入口
│   ├── main.py                  # 主程序（完整功能）
│   ├── run_ocr.py               # 简化版CLI
│   ├── ocr_gui.py               # 图形界面（标准版）
│   ├── ocr_gui_modern.py        # 图形界面（现代暗色主题）
│   └── 启动OCR工具.bat          # Windows快捷启动
│
├── 高级功能
│   ├── batch_processor.py       # 批量处理（多线程预加载）
│   ├── model_optimizer.py       # 自动参数优化
│   └── api_server.py            # Web API服务（FastAPI）
│
├── 部署配置
│   ├── Dockerfile               # Docker镜像
│   ├── docker-compose.yml       # 容器编排
│   ├── requirements.txt         # Python依赖
│   └── requirements-api.txt     # API服务依赖
│
├── 配置文件
│   ├── config.yaml              # 主配置文件
│   └── config.template.yaml     # 配置模板
│
├── 工具脚本
│   ├── test_performance.py      # 性能测试
│   ├── diagnose_gpu.py          # GPU诊断
│   └── setup_environment.ps1    # 环境配置脚本
│
└── README.md
```

## ⚙️ 配置说明

编辑 `config.yaml` 自定义设置：

```yaml
# OCR引擎选择
ocr:
  engine: "deepseek-ocr2"  # 或 "paddleocr"
  use_gpu: true

# 处理参数
processing:
  dpi: 150  # 降低DPI可提速（100-200）
  
# 匹配阈值
matching:
  similarity_threshold: 0.75
  exact_match_threshold: 0.95
```

## 📊 输出说明

处理完成后，输出目录结构：

```
输出目录/
├── 凭证_可搜索/           # 可搜索的凭证PDF
│   ├── 文件1.pdf
│   └── 文件2.pdf
├── 参照资料_可搜索/       # 可搜索的参照PDF
│   ├── 资料1.pdf
│   └── 资料2.pdf
└── 对比报告.xlsx          # Excel格式对比报告
```

对比报告包含：
- 凭证文件名和页码
- 匹配状态（匹配/部分匹配/未匹配）
- 匹配的参照文件和页码
- 相似度百分比
- 关键词匹配信息

## 🔧 性能优化

### 提高速度

1. **关闭后台程序** - 释放GPU显存
2. **降低DPI** - 在config.yaml中设置 `dpi: 150`
3. **使用缓存** - 程序自动缓存已处理的文件

### GPU显存要求

| 配置 | 最低显存 | 推荐显存 |
|------|---------|---------|
| DeepSeek-OCR2 | 8GB | 12GB |
| PaddleOCR | 2GB | 4GB |

## 🐛 常见问题

### Q: 处理速度很慢怎么办？
A: 
1. 检查GPU占用：`nvidia-smi`
2. 关闭后台程序（浏览器、模拟器等）
3. 降低DPI到150

### Q: 显存不足怎么办？
A: 
1. 关闭其他占用GPU的程序
2. 切换到PaddleOCR引擎
3. 分批处理大量文件

### Q: 中文识别不准确？
A: 
1. 提高DPI到200
2. 确保扫描件清晰度足够
3. 检查PDF是否为纯图像格式

## 📝 更新日志

### v1.0.0 (2026-01-30)
- ✅ 支持DeepSeek-OCR2和PaddleOCR双引擎
- ✅ 图形化界面
- ✅ 可搜索PDF生成
- ✅ 智能文档比对
- ✅ GPU加速支持

## 📄 许可证

MIT License

## 🙏 致谢

- [DeepSeek-OCR2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) - 高精度OCR模型
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 备选OCR引擎
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF处理库
