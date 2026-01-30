# PDF智能OCR分类与比对系统 - 环境配置脚本
# 运行方式: 在PowerShell中执行 .\setup_environment.ps1

Write-Host "========================================"
Write-Host "PDF OCR分类系统 - 环境配置"
Write-Host "========================================"

# 检查Python
Write-Host "[1/5] 检查Python版本..."
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python未安装，请先安装Python 3.9+" -ForegroundColor Red
    exit 1
}

# 创建虚拟环境
Write-Host "[2/5] 创建虚拟环境..."
if (-not (Test-Path ".\venv")) {
    python -m venv venv
    Write-Host "虚拟环境已创建" -ForegroundColor Green
}
else {
    Write-Host "虚拟环境已存在" -ForegroundColor Yellow
}

# 激活虚拟环境
Write-Host "[3/5] 激活虚拟环境..."
& ".\venv\Scripts\Activate.ps1"

# 升级pip并安装依赖
Write-Host "[4/5] 安装依赖（这需要几分钟）..."
python -m pip install --upgrade pip

Write-Host "安装PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "安装PaddlePaddle..."
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

Write-Host "安装其他依赖..."
pip install paddleocr pdf2image PyMuPDF Pillow jieba numpy opencv-python pyyaml openpyxl tqdm rich

# 检查Poppler
Write-Host "[5/5] 检查Poppler..."
$popplerPath = "C:\poppler\Library\bin"
if (Test-Path $popplerPath) {
    Write-Host "Poppler已安装: $popplerPath" -ForegroundColor Green
}
else {
    Write-Host "Poppler未找到！请下载安装:" -ForegroundColor Yellow
    Write-Host "https://github.com/oschwartz10612/poppler-windows/releases"
    Write-Host "解压到 C:\poppler"
}

# 验证
Write-Host ""
Write-Host "========================================"
Write-Host "验证安装"
Write-Host "========================================"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import paddle; print('Paddle GPU:', paddle.device.is_compiled_with_cuda())"

Write-Host ""
Write-Host "配置完成！运行: python main.py" -ForegroundColor Green
