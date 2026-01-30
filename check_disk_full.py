"""
C盘全面检查脚本
分析所有可能占用大量空间的目录
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def get_size(path):
    """获取文件夹大小"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_size(entry.path)
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return total

def format_size(size):
    """格式化大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def check_all():
    home = Path.home()
    
    print("=" * 70)
    print("C盘全面空间分析")
    print("=" * 70)
    
    # C盘总体
    c_drive = shutil.disk_usage("C:\\")
    print(f"\n【C盘概览】")
    print(f"  总计: {format_size(c_drive.total)}")
    print(f"  已用: {format_size(c_drive.used)} ({c_drive.used/c_drive.total*100:.1f}%)")
    print(f"  剩余: {format_size(c_drive.free)}")
    
    # 需要检查的目录
    check_dirs = {
        # 开发相关
        "Huggingface缓存": home / ".cache" / "huggingface",
        "PyTorch缓存": home / ".cache" / "torch",
        "pip缓存": home / "AppData" / "Local" / "pip" / "cache",
        "Conda环境": home / "anaconda3" / "envs",
        "Conda包缓存": home / "anaconda3" / "pkgs",
        "npm缓存": home / "AppData" / "Roaming" / "npm-cache",
        "yarn缓存": home / "AppData" / "Local" / "Yarn" / "Cache",
        
        # 浏览器
        "Chrome缓存": home / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
        "Edge缓存": home / "AppData" / "Local" / "Microsoft" / "Edge" / "User Data",
        
        # 系统
        "临时文件": Path(os.environ.get('TEMP', 'C:\\Temp')),
        "Windows临时": Path("C:\\Windows\\Temp"),
        "下载文件夹": home / "Downloads",
        "回收站": Path("C:\\$Recycle.Bin"),
        
        # 应用
        "微信": home / "Documents" / "WeChat Files",
        "QQ": home / "Documents" / "Tencent Files",
        "Steam": Path("C:\\Program Files (x86)\\Steam\\steamapps"),
        "Epic Games": Path("C:\\Program Files\\Epic Games"),
        "VS Code扩展": home / ".vscode" / "extensions",
        
        # 云盘
        "OneDrive": home / "OneDrive",
        "iCloud": home / "iCloudDrive",
        
        # 虚拟机
        "Docker": home / "AppData" / "Local" / "Docker",
        "WSL": home / "AppData" / "Local" / "Packages",
    }
    
    results = []
    
    for name, path in check_dirs.items():
        if path.exists():
            size = get_size(path)
            if size > 100 * 1024 * 1024:  # 大于100MB
                results.append((name, path, size))
    
    # 按大小排序
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "=" * 70)
    print("【大型目录排行榜】(大于100MB)")
    print("=" * 70)
    
    total = 0
    for name, path, size in results:
        total += size
        bar_len = int(size / (1024**3) * 5)  # 每GB 5个方块
        bar = "█" * min(bar_len, 30)
        print(f"\n{name}")
        print(f"  {bar} {format_size(size)}")
        print(f"  路径: {path}")
    
    print("\n" + "-" * 70)
    print(f"以上目录总计: {format_size(total)}")
    
    # 详细分析大目录
    print("\n" + "=" * 70)
    print("【详细分析】")
    print("=" * 70)
    
    # Huggingface模型
    hf_cache = home / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        print("\n📦 Huggingface模型:")
        for m in hf_cache.iterdir():
            if m.is_dir():
                size = get_size(m)
                if size > 50 * 1024 * 1024:
                    print(f"  • {m.name}: {format_size(size)}")
    
    # Conda环境
    conda_envs = home / "anaconda3" / "envs"
    if conda_envs.exists():
        print("\n🐍 Conda环境:")
        for env in conda_envs.iterdir():
            if env.is_dir():
                size = get_size(env)
                if size > 100 * 1024 * 1024:
                    print(f"  • {env.name}: {format_size(size)}")
    
    # 下载文件夹大文件
    downloads = home / "Downloads"
    if downloads.exists():
        print("\n📥 下载文件夹大文件:")
        files = []
        for f in downloads.iterdir():
            if f.is_file():
                try:
                    size = f.stat().st_size
                    if size > 100 * 1024 * 1024:
                        files.append((f.name, size))
                except:
                    pass
        files.sort(key=lambda x: x[1], reverse=True)
        for name, size in files[:10]:
            print(f"  • {name}: {format_size(size)}")
    
    # 清理命令汇总
    print("\n" + "=" * 70)
    print("【清理命令汇总】")
    print("=" * 70)
    
    commands = [
        ("pip缓存", "pip cache purge", "删除已下载的pip包缓存"),
        ("Conda缓存", "conda clean --all -y", "删除Conda包缓存和未使用的包"),
        ("临时文件", "rd /s /q %TEMP%\\*", "删除用户临时文件"),
        ("系统临时", "管理员运行: rd /s /q C:\\Windows\\Temp\\*", "删除系统临时文件"),
        ("npm缓存", "npm cache clean --force", "删除npm包缓存"),
        ("Docker", "docker system prune -a", "删除无用的Docker镜像和容器"),
    ]
    
    for name, cmd, desc in commands:
        print(f"\n{name}:")
        print(f"  命令: {cmd}")
        print(f"  说明: {desc}")
    
    print("\n" + "=" * 70)
    print("【可安全删除的建议】")
    print("=" * 70)
    print("""
1. ✅ pip缓存 (4.55 GB) - 可以完全删除，需要时会重新下载
2. ✅ Conda包缓存 - 可以删除，保留已安装的环境
3. ✅ 临时文件 - 可以删除
4. ⚠️ 下载文件夹 - 检查后手动删除不需要的
5. ⚠️ 旧的conda环境 - 确认不需要后删除
6. ❌ Huggingface模型缓存 - 删除后需重新下载(6GB)
""")

if __name__ == "__main__":
    check_all()
