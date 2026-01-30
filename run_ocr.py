"""
PDF OCR处理工具 - 简化封装
=============================

功能：
1. 对两个文件夹中的PDF进行OCR识别
2. 生成可搜索的PDF文件
3. 生成内容对比报告

用法：
    python run_ocr.py <凭证文件夹> <参照资料文件夹> [输出文件夹]

示例：
    python run_ocr.py "C:/凭证" "C:/扫描资料" "C:/输出结果"
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_pdfs(folder: str) -> List[Path]:
    """递归查找文件夹中的所有PDF"""
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error(f"文件夹不存在: {folder}")
        return []
    
    pdfs = list(folder_path.rglob("*.pdf"))
    logger.info(f"在 {folder} 中找到 {len(pdfs)} 个PDF文件")
    return pdfs


def create_searchable_pdf(input_pdf: Path, output_pdf: Path, ocr_text: str) -> bool:
    """
    创建可搜索的PDF（将OCR文本嵌入PDF作为隐藏文本层）
    
    使用PyMuPDF将OCR文本添加到PDF中
    """
    try:
        import fitz  # PyMuPDF
        
        # 打开原始PDF
        doc = fitz.open(str(input_pdf))
        
        # 解析OCR文本（按页分割）
        pages_text = ocr_text.split("\n=== 第")
        page_texts = {}
        
        for part in pages_text:
            if "页 ===" in part:
                try:
                    page_num = int(part.split("页 ===")[0].strip())
                    text = part.split("===\n", 1)[1] if "===\n" in part else ""
                    page_texts[page_num] = text
                except:
                    pass
        
        # 为每页添加隐藏文本层
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page_texts.get(page_num + 1, "")
            
            if text:
                # 在页面上添加不可见的文本（用于搜索）
                # 使用白色文字覆盖在页面上，实际看不见但可以搜索
                rect = page.rect
                # 添加文本到页面（设置为透明/不可见）
                text_point = fitz.Point(0, 0)
                page.insert_text(
                    text_point,
                    text,
                    fontsize=1,  # 极小字体
                    color=(1, 1, 1),  # 白色（不可见）
                    render_mode=3  # 不可见模式
                )
        
        # 保存带文本层的PDF
        doc.save(str(output_pdf))
        doc.close()
        
        return True
        
    except ImportError:
        # 如果没有PyMuPDF，使用备选方案
        logger.warning("PyMuPDF未安装，使用备选方案（复制PDF+保存txt）")
        try:
            shutil.copy2(input_pdf, output_pdf)
            txt_path = output_pdf.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            return True
        except Exception as e:
            logger.error(f"创建可搜索PDF失败: {e}")
            return False
            
    except Exception as e:
        logger.error(f"创建可搜索PDF失败: {e}")
        return False


def run_ocr_pipeline(
    voucher_folder: str,
    reference_folder: str, 
    output_folder: str
) -> Dict:
    """
    运行完整的OCR处理流程
    
    Args:
        voucher_folder: 凭证文件夹路径
        reference_folder: 参照资料文件夹路径
        output_folder: 输出文件夹路径
        
    Returns:
        处理结果统计
    """
    from pdf_processor import PDFProcessor
    from deepseek_ocr2_engine import DeepSeekOCR2Engine
    from content_matcher import ContentMatcher, PageFeatures
    from ocr_engine import OCRResultExtractor
    from progress_display import get_progress_display
    
    # 创建输出目录
    output_path = Path(output_folder)
    voucher_output = output_path / "凭证_可搜索"
    reference_output = output_path / "参照资料_可搜索"
    report_path = output_path / "对比报告.xlsx"
    
    voucher_output.mkdir(parents=True, exist_ok=True)
    reference_output.mkdir(parents=True, exist_ok=True)
    
    # 初始化组件
    logger.info("初始化OCR引擎...")
    pdf_processor = PDFProcessor({'dpi': 150})  # 使用150 DPI提高速度
    ocr_engine = DeepSeekOCR2Engine({})
    feature_extractor = OCRResultExtractor()
    matcher = ContentMatcher({
        'similarity_threshold': 0.75,
        'exact_match_threshold': 0.95
    })
    
    # 查找PDF文件
    voucher_pdfs = find_pdfs(voucher_folder)
    reference_pdfs = find_pdfs(reference_folder)
    
    total_files = len(voucher_pdfs) + len(reference_pdfs)
    if total_files == 0:
        logger.warning("未找到任何PDF文件")
        return {'status': 'no_files'}
    
    # 处理结果
    all_voucher_pages: List[PageFeatures] = []
    all_reference_pages: List[PageFeatures] = []
    
    progress = get_progress_display(True)
    progress.start(total_files)
    
    def process_pdf_folder(
        pdfs: List[Path], 
        output_dir: Path,
        pages_list: List[PageFeatures],
        folder_type: str
    ) -> int:
        """处理一个文件夹中的所有PDF"""
        processed_pages = 0
        
        for pdf_file in pdfs:
            progress.update_file(pdf_file.name)
            
            try:
                # 获取相对路径用于保存
                rel_path = pdf_file.name
                output_pdf = output_dir / rel_path
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                
                # OCR处理
                all_text = []
                page_num = 0
                
                for page_num, page_image in pdf_processor.process_pdf(str(pdf_file)):
                    # OCR识别
                    ocr_result = ocr_engine.recognize_pdf_page(page_image, page_num)
                    page_text = ocr_result.get_full_text()
                    all_text.append(f"=== 第{page_num}页 ===\n{page_text}")
                    
                    # 提取特征
                    features = feature_extractor.extract_features(page_text)
                    page_features = PageFeatures(
                        file_path=str(pdf_file),
                        page_num=page_num,
                        text=page_text,
                        doc_type="未分类",
                        dates=features.get('dates', []),
                        amounts=features.get('amounts', []),
                        numbers=features.get('numbers', []),
                        keywords=features.get('keywords', [])
                    )
                    pages_list.append(page_features)
                    processed_pages += 1
                
                # 保存可搜索PDF
                full_text = "\n\n".join(all_text)
                create_searchable_pdf(pdf_file, output_pdf, full_text)
                
                progress.complete_file()
                logger.info(f"已处理: {pdf_file.name} ({page_num}页)")
                
            except Exception as e:
                logger.error(f"处理失败 {pdf_file}: {e}")
                progress.complete_file()
        
        return processed_pages
    
    try:
        # 处理参照资料
        logger.info("\n处理参照资料...")
        ref_pages = process_pdf_folder(
            reference_pdfs, reference_output, all_reference_pages, "参照"
        )
        
        # 构建参照索引
        matcher.build_reference_index(all_reference_pages)
        
        # 处理凭证
        logger.info("\n处理凭证文件...")
        voucher_pages = process_pdf_folder(
            voucher_pdfs, voucher_output, all_voucher_pages, "凭证"
        )
        
    finally:
        progress.stop()
    
    # 生成对比报告
    logger.info("\n生成对比报告...")
    match_results = []
    
    for page in all_voucher_pages:
        matches = matcher.find_matches(page)
        if matches:
            best_match = matches[0]
            match_results.append({
                '凭证文件': Path(page.file_path).name,
                '凭证页码': page.page_num,
                '匹配状态': '匹配' if best_match[1] > 0.75 else '部分匹配',
                '参照文件': Path(best_match[0].file_path).name,
                '参照页码': best_match[0].page_num,
                '相似度': f"{best_match[1]:.2%}",
                '凭证关键词': ', '.join(page.keywords[:5]),
                '参照关键词': ', '.join(best_match[0].keywords[:5])
            })
        else:
            match_results.append({
                '凭证文件': Path(page.file_path).name,
                '凭证页码': page.page_num,
                '匹配状态': '未匹配',
                '参照文件': '-',
                '参照页码': '-',
                '相似度': '-',
                '凭证关键词': ', '.join(page.keywords[:5]),
                '参照关键词': '-'
            })
    
    # 保存Excel报告
    try:
        import pandas as pd
        df = pd.DataFrame(match_results)
        df.to_excel(report_path, index=False, sheet_name='对比结果')
        logger.info(f"对比报告已保存: {report_path}")
    except ImportError:
        # 如果没有pandas，保存为CSV
        import csv
        csv_path = output_path / "对比报告.csv"
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            if match_results:
                writer = csv.DictWriter(f, fieldnames=match_results[0].keys())
                writer.writeheader()
                writer.writerows(match_results)
        logger.info(f"对比报告已保存: {csv_path}")
    
    # 返回统计
    stats = {
        'status': 'success',
        'voucher_files': len(voucher_pdfs),
        'voucher_pages': voucher_pages,
        'reference_files': len(reference_pdfs),
        'reference_pages': ref_pages,
        'matched': sum(1 for r in match_results if r['匹配状态'] == '匹配'),
        'partial': sum(1 for r in match_results if r['匹配状态'] == '部分匹配'),
        'unmatched': sum(1 for r in match_results if r['匹配状态'] == '未匹配'),
        'output_folder': str(output_path),
        'report_path': str(report_path)
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='PDF OCR处理工具 - 生成可搜索PDF和对比报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python run_ocr.py "C:/凭证" "C:/扫描资料"
    python run_ocr.py "C:/凭证" "C:/扫描资料" "C:/输出结果"
        '''
    )
    
    parser.add_argument('voucher_folder', help='凭证文件夹路径')
    parser.add_argument('reference_folder', help='参照资料文件夹路径')
    parser.add_argument('output_folder', nargs='?', default=None,
                        help='输出文件夹路径（默认为当前目录下的 OCR_结果_时间戳）')
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if args.output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_folder = f"OCR_结果_{timestamp}"
    
    # 验证输入路径
    if not os.path.isdir(args.voucher_folder):
        logger.error(f"凭证文件夹不存在: {args.voucher_folder}")
        sys.exit(1)
    
    if not os.path.isdir(args.reference_folder):
        logger.error(f"参照资料文件夹不存在: {args.reference_folder}")
        sys.exit(1)
    
    # 打印配置
    print("\n" + "="*60)
    print("PDF OCR处理工具")
    print("="*60)
    print(f"凭证文件夹:     {args.voucher_folder}")
    print(f"参照资料文件夹: {args.reference_folder}")
    print(f"输出文件夹:     {args.output_folder}")
    print("="*60 + "\n")
    
    # 运行处理
    try:
        stats = run_ocr_pipeline(
            args.voucher_folder,
            args.reference_folder,
            args.output_folder
        )
        
        if stats['status'] == 'success':
            print("\n" + "="*60)
            print("✅ 处理完成!")
            print("="*60)
            print(f"凭证文件: {stats['voucher_files']} 个 ({stats['voucher_pages']} 页)")
            print(f"参照文件: {stats['reference_files']} 个 ({stats['reference_pages']} 页)")
            print("-"*60)
            print(f"匹配:     {stats['matched']} 页")
            print(f"部分匹配: {stats['partial']} 页")
            print(f"未匹配:   {stats['unmatched']} 页")
            print("-"*60)
            print(f"输出目录: {stats['output_folder']}")
            print(f"对比报告: {stats['report_path']}")
            print("="*60)
        else:
            print("\n⚠️ 未找到PDF文件")
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


def run_ocr_pipeline_with_callback(
    voucher_folder: str,
    reference_folder: str, 
    output_folder: str,
    callback=None
) -> Dict:
    """
    带回调的OCR处理流程（用于GUI）
    
    Args:
        voucher_folder: 凭证文件夹路径
        reference_folder: 参照资料文件夹路径
        output_folder: 输出文件夹路径
        callback: 进度回调函数 callback(msg_type, **kwargs)
        
    Returns:
        处理结果统计
    """
    from pdf_processor import PDFProcessor
    from deepseek_ocr2_engine import DeepSeekOCR2Engine
    from content_matcher import ContentMatcher, PageFeatures
    from ocr_engine import OCRResultExtractor
    
    def notify(msg_type, **kwargs):
        if callback:
            callback(msg_type, **kwargs)
    
    # 创建输出目录
    output_path = Path(output_folder)
    voucher_output = output_path / "凭证_可搜索"
    reference_output = output_path / "参照资料_可搜索"
    report_path = output_path / "对比报告.xlsx"
    
    voucher_output.mkdir(parents=True, exist_ok=True)
    reference_output.mkdir(parents=True, exist_ok=True)
    
    # 初始化组件
    notify('status', text='初始化OCR引擎...')
    notify('log', text='初始化OCR引擎...')
    
    pdf_processor = PDFProcessor({'dpi': 150})
    ocr_engine = DeepSeekOCR2Engine({})
    feature_extractor = OCRResultExtractor()
    matcher = ContentMatcher({
        'similarity_threshold': 0.75,
        'exact_match_threshold': 0.95
    })
    
    # 查找PDF文件
    voucher_pdfs = find_pdfs(voucher_folder)
    reference_pdfs = find_pdfs(reference_folder)
    
    total_files = len(voucher_pdfs) + len(reference_pdfs)
    if total_files == 0:
        notify('log', text='未找到任何PDF文件')
        return {'status': 'no_files'}
    
    notify('log', text=f'找到 {len(voucher_pdfs)} 个凭证文件, {len(reference_pdfs)} 个参照文件')
    
    # 处理结果
    all_voucher_pages: List[PageFeatures] = []
    all_reference_pages: List[PageFeatures] = []
    processed_files = 0
    
    def process_pdf_folder(
        pdfs: List[Path], 
        output_dir: Path,
        pages_list: List[PageFeatures],
        folder_type: str
    ) -> int:
        nonlocal processed_files
        processed_pages = 0
        
        for pdf_file in pdfs:
            notify('file', text=pdf_file.name)
            notify('status', text=f'处理{folder_type}: {pdf_file.name}')
            notify('log', text=f'处理: {pdf_file.name}')
            
            try:
                # 获取相对路径用于保存
                rel_path = pdf_file.name
                output_pdf = output_dir / rel_path
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                
                # OCR处理
                all_text = []
                page_num = 0
                
                for page_num, page_image in pdf_processor.process_pdf(str(pdf_file)):
                    # OCR识别
                    ocr_result = ocr_engine.recognize_pdf_page(page_image, page_num)
                    page_text = ocr_result.get_full_text()
                    all_text.append(f"=== 第{page_num}页 ===\n{page_text}")
                    
                    # 提取特征
                    features = feature_extractor.extract_features(page_text)
                    page_features = PageFeatures(
                        file_path=str(pdf_file),
                        page_num=page_num,
                        text=page_text,
                        doc_type="未分类",
                        dates=features.get('dates', []),
                        amounts=features.get('amounts', []),
                        numbers=features.get('numbers', []),
                        keywords=features.get('keywords', [])
                    )
                    pages_list.append(page_features)
                    processed_pages += 1
                
                # 保存可搜索PDF
                full_text = "\n\n".join(all_text)
                create_searchable_pdf(pdf_file, output_pdf, full_text)
                
                processed_files += 1
                progress = (processed_files / total_files) * 100
                notify('progress', value=progress)
                notify('log', text=f'完成: {pdf_file.name} ({page_num}页)')
                
            except Exception as e:
                notify('log', text=f'错误: {pdf_file.name} - {e}')
                processed_files += 1
        
        return processed_pages
    
    # 处理参照资料
    notify('status', text='处理参照资料...')
    notify('log', text='开始处理参照资料...')
    ref_pages = process_pdf_folder(
        reference_pdfs, reference_output, all_reference_pages, "参照"
    )
    
    # 构建参照索引
    matcher.build_reference_index(all_reference_pages)
    
    # 处理凭证
    notify('status', text='处理凭证文件...')
    notify('log', text='开始处理凭证文件...')
    voucher_pages = process_pdf_folder(
        voucher_pdfs, voucher_output, all_voucher_pages, "凭证"
    )
    
    # 生成对比报告
    notify('status', text='生成对比报告...')
    notify('log', text='生成对比报告...')
    match_results = []
    
    for page in all_voucher_pages:
        matches = matcher.find_matches(page)
        if matches:
            best_match = matches[0]
            match_results.append({
                '凭证文件': Path(page.file_path).name,
                '凭证页码': page.page_num,
                '匹配状态': '匹配' if best_match[1] > 0.75 else '部分匹配',
                '参照文件': Path(best_match[0].file_path).name,
                '参照页码': best_match[0].page_num,
                '相似度': f"{best_match[1]:.2%}",
                '凭证关键词': ', '.join(page.keywords[:5]),
                '参照关键词': ', '.join(best_match[0].keywords[:5])
            })
        else:
            match_results.append({
                '凭证文件': Path(page.file_path).name,
                '凭证页码': page.page_num,
                '匹配状态': '未匹配',
                '参照文件': '-',
                '参照页码': '-',
                '相似度': '-',
                '凭证关键词': ', '.join(page.keywords[:5]),
                '参照关键词': '-'
            })
    
    # 保存Excel报告
    try:
        import pandas as pd
        df = pd.DataFrame(match_results)
        df.to_excel(report_path, index=False, sheet_name='对比结果')
        notify('log', text=f'对比报告已保存: {report_path}')
    except ImportError:
        import csv
        csv_path = output_path / "对比报告.csv"
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            if match_results:
                writer = csv.DictWriter(f, fieldnames=match_results[0].keys())
                writer.writeheader()
                writer.writerows(match_results)
        notify('log', text=f'对比报告已保存: {csv_path}')
    
    # 返回统计
    stats = {
        'status': 'success',
        'voucher_files': len(voucher_pdfs),
        'voucher_pages': voucher_pages,
        'reference_files': len(reference_pdfs),
        'reference_pages': ref_pages,
        'matched': sum(1 for r in match_results if r['匹配状态'] == '匹配'),
        'partial': sum(1 for r in match_results if r['匹配状态'] == '部分匹配'),
        'unmatched': sum(1 for r in match_results if r['匹配状态'] == '未匹配'),
        'output_folder': str(output_path),
        'report_path': str(report_path)
    }
    
    return stats
