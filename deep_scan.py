"""
C盘深度分析 - 找出所有大型目录
"""

import os
from pathlib import Path
from collections import defaultdict

def get_size_fast(path):
    """快速获取文件夹大小"""
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat().st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += get_size_fast(entry.path)
                except (PermissionError, OSError):
                    pass
    except (PermissionError, OSError):
        pass
    return total

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

print("=" * 70)
print("C盘深度扫描（请稍候，这需要几分钟...）")
print("=" * 70)

home = Path.home()

# 扫描所有可能的大目录
scan_roots = [
    # 用户目录下所有顶级文件夹
    home,
    # Program Files
    Path("C:\\Program Files"),
    Path("C:\\Program Files (x86)"),
    # ProgramData
    Path("C:\\ProgramData"),
    # Windows用户数据
    Path("C:\\Users"),
]

big_dirs = []

# 扫描用户目录的每个子文件夹
print(f"\n扫描用户目录: {home}")
for item in home.iterdir():
    if item.is_dir():
        try:
            size = get_size_fast(str(item))
            if size > 500 * 1024 * 1024:  # 大于500MB
                big_dirs.append((str(item), size))
                print(f"  {item.name}: {format_size(size)}")
        except:
            pass

# 扫描AppData内部
appdata_local = home / "AppData" / "Local"
if appdata_local.exists():
    print(f"\n扫描AppData\\Local...")
    for item in appdata_local.iterdir():
        if item.is_dir():
            try:
                size = get_size_fast(str(item))
                if size > 300 * 1024 * 1024:  # 大于300MB
                    big_dirs.append((str(item), size))
                    print(f"  {item.name}: {format_size(size)}")
            except:
                pass

appdata_roaming = home / "AppData" / "Roaming"
if appdata_roaming.exists():
    print(f"\n扫描AppData\\Roaming...")
    for item in appdata_roaming.iterdir():
        if item.is_dir():
            try:
                size = get_size_fast(str(item))
                if size > 300 * 1024 * 1024:
                    big_dirs.append((str(item), size))
                    print(f"  {item.name}: {format_size(size)}")
            except:
                pass

# 扫描Program Files
print(f"\n扫描Program Files...")
for root in [Path("C:\\Program Files"), Path("C:\\Program Files (x86)")]:
    if root.exists():
        for item in root.iterdir():
            if item.is_dir():
                try:
                    size = get_size_fast(str(item))
                    if size > 500 * 1024 * 1024:
                        big_dirs.append((str(item), size))
                        print(f"  {item.name}: {format_size(size)}")
                except:
                    pass

# 扫描其他常见大目录
other_dirs = [
    Path("C:\\ProgramData"),
    Path("C:\\Windows\\Installer"),
    Path("C:\\Windows\\WinSxS"),
    home / ".nuget",
    home / ".gradle",
    home / ".m2",
    home / ".npm",
    home / "scoop",
    home / "go",
    home / ".rustup",
    home / ".cargo",
]

print(f"\n扫描其他常见目录...")
for d in other_dirs:
    if d.exists():
        try:
            size = get_size_fast(str(d))
            if size > 300 * 1024 * 1024:
                big_dirs.append((str(d), size))
                print(f"  {d.name}: {format_size(size)}")
        except:
            pass

# 去重和排序
unique_dirs = {}
for path, size in big_dirs:
    # 避免重复计算（父目录和子目录都在列表中）
    is_child = False
    for existing in list(unique_dirs.keys()):
        if path.startswith(existing + "\\"):
            is_child = True
            break
        if existing.startswith(path + "\\"):
            del unique_dirs[existing]
    if not is_child:
        unique_dirs[path] = size

sorted_dirs = sorted(unique_dirs.items(), key=lambda x: x[1], reverse=True)

# 输出结果
print("\n" + "=" * 70)
print("【C盘空间占用排行榜】")
print("=" * 70)

total_found = 0
for path, size in sorted_dirs[:30]:  # 显示前30个
    total_found += size
    bar_len = int(size / (1024**3) * 3)
    bar = "█" * min(bar_len, 30)
    name = Path(path).name
    print(f"\n{name}")
    print(f"  {bar} {format_size(size)}")
    print(f"  {path}")

print("\n" + "-" * 70)
print(f"已定位总计: {format_size(total_found)}")

# 计算未知空间
import shutil
c_drive = shutil.disk_usage("C:\\")
unknown = c_drive.used - total_found
print(f"C盘已用: {format_size(c_drive.used)}")
print(f"未定位: {format_size(unknown)}")

print("\n" + "=" * 70)
print("【未定位空间可能在】")
print("=" * 70)
print("""
• Windows系统文件 (C:\\Windows)
• 系统还原点
• 休眠文件 (hiberfil.sys)
• 页面文件 (pagefile.sys)
• 虚拟内存

使用Windows磁盘清理工具清理系统文件:
  右键C盘 -> 属性 -> 磁盘清理 -> 清理系统文件
""")
