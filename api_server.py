"""
PDF OCR处理工具 - Web API服务
==============================

提供RESTful API接口，支持远程调用OCR处理服务
"""

import os
import sys
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 添加当前目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# 数据模型
# ============================================

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OCREngine(str, Enum):
    """OCR引擎"""
    DEEPSEEK = "deepseek-ocr2"
    PADDLE = "paddleocr"


class TaskCreateRequest(BaseModel):
    """创建任务请求"""
    engine: OCREngine = OCREngine.DEEPSEEK
    dpi: int = Field(default=150, ge=72, le=600)
    generate_searchable_pdf: bool = True
    compare_documents: bool = True


class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    total_files: int = 0
    processed_files: int = 0
    total_pages: int = 0
    processed_pages: int = 0
    error_message: Optional[str] = None
    result_url: Optional[str] = None


class OCRResult(BaseModel):
    """OCR结果"""
    file_name: str
    page_count: int
    text_preview: str
    processing_time: float


class ComparisonResult(BaseModel):
    """比对结果"""
    voucher_file: str
    reference_file: str
    similarity: float
    match_status: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used: Optional[float] = None
    version: str = "1.0.0"


# ============================================
# 任务管理器
# ============================================

class TaskManager:
    """任务管理器"""
    
    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_data: Dict[str, Dict] = {}
        
        # 创建必要的目录
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "temp").mkdir(parents=True, exist_ok=True)
    
    def create_task(self, config: TaskCreateRequest) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())[:8]
        
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        self.tasks[task_id] = task_info
        self.task_data[task_id] = {
            'config': config.dict(),
            'voucher_files': [],
            'reference_files': [],
            'results': []
        }
        
        # 创建任务目录
        task_dir = self.data_dir / "uploads" / task_id
        (task_dir / "vouchers").mkdir(parents=True, exist_ok=True)
        (task_dir / "references").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created task: {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
    
    def add_file(self, task_id: str, file_type: str, file_path: str):
        """添加文件到任务"""
        if task_id in self.task_data:
            key = f"{file_type}_files"
            if key in self.task_data[task_id]:
                self.task_data[task_id][key].append(file_path)
    
    def get_task_dir(self, task_id: str) -> Path:
        """获取任务目录"""
        return self.data_dir / "uploads" / task_id
    
    def get_result_dir(self, task_id: str) -> Path:
        """获取结果目录"""
        result_dir = self.data_dir / "results" / task_id
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir


# 全局任务管理器
task_manager = TaskManager()


# ============================================
# FastAPI应用
# ============================================

app = FastAPI(
    title="PDF OCR处理服务",
    description="智能PDF OCR识别与文档比对API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API端点
# ============================================

@app.get("/", tags=["General"])
async def root():
    """根路径"""
    return {
        "service": "PDF OCR处理服务",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """健康检查"""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_used = None
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
    
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used=gpu_memory_used
    )


@app.post("/tasks", response_model=TaskInfo, tags=["Tasks"])
async def create_task(config: TaskCreateRequest = None):
    """创建OCR处理任务"""
    if config is None:
        config = TaskCreateRequest()
    
    task_id = task_manager.create_task(config)
    return task_manager.get_task(task_id)


@app.get("/tasks/{task_id}", response_model=TaskInfo, tags=["Tasks"])
async def get_task(task_id: str):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.post("/tasks/{task_id}/vouchers", tags=["Files"])
async def upload_voucher(
    task_id: str,
    files: List[UploadFile] = File(...)
):
    """上传凭证文件"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="任务已开始处理，无法上传文件")
    
    task_dir = task_manager.get_task_dir(task_id) / "vouchers"
    uploaded = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
        
        file_path = task_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        task_manager.add_file(task_id, "voucher", str(file_path))
        uploaded.append(file.filename)
    
    task_manager.update_task(
        task_id,
        total_files=len(task_manager.task_data[task_id]['voucher_files']) +
                   len(task_manager.task_data[task_id]['reference_files'])
    )
    
    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/tasks/{task_id}/references", tags=["Files"])
async def upload_reference(
    task_id: str,
    files: List[UploadFile] = File(...)
):
    """上传参照资料文件"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="任务已开始处理，无法上传文件")
    
    task_dir = task_manager.get_task_dir(task_id) / "references"
    uploaded = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
        
        file_path = task_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        task_manager.add_file(task_id, "reference", str(file_path))
        uploaded.append(file.filename)
    
    task_manager.update_task(
        task_id,
        total_files=len(task_manager.task_data[task_id]['voucher_files']) +
                   len(task_manager.task_data[task_id]['reference_files'])
    )
    
    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/tasks/{task_id}/start", response_model=TaskInfo, tags=["Tasks"])
async def start_task(task_id: str, background_tasks: BackgroundTasks):
    """开始处理任务"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="任务已开始或已完成")
    
    # 检查是否有文件
    task_data = task_manager.task_data[task_id]
    if not task_data['voucher_files'] and not task_data['reference_files']:
        raise HTTPException(status_code=400, detail="请先上传文件")
    
    # 更新状态并启动后台任务
    task_manager.update_task(
        task_id,
        status=TaskStatus.PROCESSING,
        started_at=datetime.now().isoformat()
    )
    
    background_tasks.add_task(process_task, task_id)
    
    return task_manager.get_task(task_id)


@app.post("/tasks/{task_id}/cancel", response_model=TaskInfo, tags=["Tasks"])
async def cancel_task(task_id: str):
    """取消任务"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status == TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务已完成，无法取消")
    
    task_manager.update_task(task_id, status=TaskStatus.CANCELLED)
    
    return task_manager.get_task(task_id)


@app.get("/tasks/{task_id}/results", tags=["Results"])
async def get_results(task_id: str):
    """获取处理结果"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")
    
    result_dir = task_manager.get_result_dir(task_id)
    
    # 列出所有结果文件
    files = []
    for f in result_dir.iterdir():
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "download_url": f"/tasks/{task_id}/download/{f.name}"
        })
    
    return {"files": files}


@app.get("/tasks/{task_id}/download/{filename}", tags=["Results"])
async def download_result(task_id: str, filename: str):
    """下载结果文件"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    file_path = task_manager.get_result_dir(task_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


# ============================================
# 后台任务处理
# ============================================

async def process_task(task_id: str):
    """后台处理任务"""
    try:
        from run_ocr import run_ocr_pipeline_with_callback
        
        task_data = task_manager.task_data[task_id]
        task_dir = task_manager.get_task_dir(task_id)
        result_dir = task_manager.get_result_dir(task_id)
        
        def progress_callback(msg_type, **kwargs):
            task = task_manager.get_task(task_id)
            if task.status == TaskStatus.CANCELLED:
                raise InterruptedError("任务已取消")
            
            if msg_type == 'progress':
                task_manager.update_task(task_id, progress=kwargs.get('value', 0))
            elif msg_type == 'file':
                pass  # 可以记录当前处理的文件
        
        # 执行处理
        stats = run_ocr_pipeline_with_callback(
            str(task_dir / "vouchers"),
            str(task_dir / "references"),
            str(result_dir),
            progress_callback
        )
        
        # 更新任务状态
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            progress=100.0,
            processed_files=stats.get('voucher_files', 0) + stats.get('reference_files', 0),
            total_pages=stats.get('voucher_pages', 0) + stats.get('reference_pages', 0),
            processed_pages=stats.get('voucher_pages', 0) + stats.get('reference_pages', 0),
            result_url=f"/tasks/{task_id}/results"
        )
        
        logger.info(f"Task {task_id} completed successfully")
        
    except InterruptedError:
        task_manager.update_task(
            task_id,
            status=TaskStatus.CANCELLED,
            completed_at=datetime.now().isoformat()
        )
        logger.info(f"Task {task_id} cancelled")
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error_message=str(e)
        )
        logger.error(f"Task {task_id} failed: {e}")


# ============================================
# 启动服务
# ============================================

def main():
    """启动API服务"""
    print("=" * 60)
    print("PDF OCR处理服务")
    print("=" * 60)
    print("API文档: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # GPU模型需要单Worker
    )


if __name__ == "__main__":
    main()
