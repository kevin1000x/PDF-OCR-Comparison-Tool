# DeepSeek-OCR2 安装脚本
# 在新的conda环境中安装

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DeepSeek-OCR2 Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 创建新的conda环境
Write-Host "[1/5] Creating conda environment (Python 3.12)..." -ForegroundColor Yellow
conda create -n deepseek-ocr2 python=3.12.9 -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create conda environment" -ForegroundColor Red
    exit 1
}

# 2. 激活环境
Write-Host "[2/5] Activating environment..." -ForegroundColor Yellow
conda activate deepseek-ocr2

# 3. 安装PyTorch (CUDA 11.8版本，可以在CUDA 13.1上运行)
Write-Host "[3/5] Installing PyTorch with CUDA 11.8..." -ForegroundColor Yellow
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装transformers和其他依赖
Write-Host "[4/5] Installing transformers and dependencies..." -ForegroundColor Yellow
pip install transformers accelerate pillow numpy

# 5. 测试
Write-Host "[5/5] Testing installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: conda activate deepseek-ocr2"
Write-Host "2. Run: python test_deepseek_ocr2.py"
