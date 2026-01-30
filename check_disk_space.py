"""
磁盘空间检查脚本
检查项目相关的大型文件和缓存
"""

import os
import shutil
from pathlib import Path

def get_size(path):
    """获取文件夹大小"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_size(entry.path)
    except PermissionError:
        pass
    return total

def format_size(size):
    """格式化大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def check_disk_space():
    """检查磁盘空间"""
    print("=" * 60)
    print("磁盘空间检查")
    print("=" * 60)
    
    # C盘总体情况
    c_drive = shutil.disk_usage("C:\\")
    print(f"\nC盘空间:")
    print(f"  总计: {format_size(c_drive.total)}")
    print(f"  已用: {format_size(c_drive.used)} ({c_drive.used/c_drive.total*100:.1f}%)")
    print(f"  剩余: {format_size(c_drive.free)} ({c_drive.free/c_drive.total*100:.1f}%)")
    
    # 检查可能占用大量空间的目录
    paths_to_check = [
        # Huggingface模型缓存
        Path.home() / ".cache" / "huggingface",
        # Conda环境
        Path.home() / "anaconda3" / "envs",
        Path.home() / "miniconda3" / "envs",
        # pip缓存
        Path.home() / "AppData" / "Local" / "pip" / "cache",
        # torch缓存
        Path.home() / ".cache" / "torch",
        # 临时文件
        Path(os.environ.get('TEMP', 'C:\\Temp')),
    ]
    
    print("\n" + "-" * 60)
    print("大型缓存目录:")
    print("-" * 60)
    
    total_cache = 0
    for path in paths_to_check:
        if path.exists():
            size = get_size(path)
            total_cache += size
            print(f"  {path}: {format_size(size)}")
    
    print(f"\n缓存总计: {format_size(total_cache)}")
    
    # 检查Huggingface具体模型
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        print("\n" + "-" * 60)
        print("Huggingface模型缓存详情:")
        print("-" * 60)
        
        for model_dir in hf_cache.iterdir():
            if model_dir.is_dir():
                size = get_size(model_dir)
                if size > 100 * 1024 * 1024:  # 大于100MB
                    print(f"  {model_dir.name}: {format_size(size)}")
    
    # 检查conda环境
    conda_envs = Path.home() / "anaconda3" / "envs"
    if conda_envs.exists():
        print("\n" + "-" * 60)
        print("Conda环境详情:")
        print("-" * 60)
        
        for env_dir in conda_envs.iterdir():
            if env_dir.is_dir():
                size = get_size(env_dir)
                if size > 500 * 1024 * 1024:  # 大于500MB
                    print(f"  {env_dir.name}: {format_size(size)}")
    
    # 项目目录
    print("\n" + "-" * 60)
    print("项目目录 (E:\\frame analysis):")
    print("-" * 60)
    
    project_dir = Path("E:\\frame analysis")
    if project_dir.exists():
        for item in project_dir.iterdir():
            if item.is_dir():
                size = get_size(item)
                if size > 10 * 1024 * 1024:  # 大于10MB
                    print(f"  {item.name}/: {format_size(size)}")
    
    # 清理建议
    print("\n" + "=" * 60)
    print("清理建议:")
    print("=" * 60)
    print("1. 删除不需要的Huggingface模型缓存:")
    print("   rm -rf ~/.cache/huggingface/hub/models--xxx")
    print()
    print("2. 清理pip缓存:")
    print("   pip cache purge")
    print()
    print("3. 清理conda缓存:")
    print("   conda clean --all")
    print()
    print("4. 删除不用的conda环境:")
    print("   conda env remove -n env_name")

if __name__ == "__main__":
    check_disk_space()
