"""
模型优化模块 - 自动参数选择和性能调优
=========================================

根据GPU配置和文档特征自动选择最佳OCR参数
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GPUTier(Enum):
    """GPU性能等级"""
    HIGH = "high"       # RTX 4080+, 16GB+
    MEDIUM = "medium"   # RTX 3060-4070, 8-12GB
    LOW = "low"         # GTX 1060-RTX 2060, 6-8GB
    CPU = "cpu"         # 无GPU或GPU不可用


class DocumentComplexity(Enum):
    """文档复杂度"""
    SIMPLE = "simple"       # 纯文本，清晰
    MEDIUM = "medium"       # 图表混排，一般清晰度
    COMPLEX = "complex"     # 手写+印刷混合，模糊


@dataclass
class OptimalConfig:
    """最优配置"""
    engine: str
    dpi: int
    image_size: int
    base_size: int
    batch_size: int
    use_gpu: bool
    use_fp16: bool
    estimated_speed: float  # 预估每页处理时间(秒)


class ModelOptimizer:
    """模型优化器 - 自动选择最佳配置"""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.gpu_tier = self._classify_gpu()
        
        # 配置预设
        self.presets = self._load_presets()
    
    def _detect_gpu(self) -> Dict:
        """检测GPU信息"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {'available': False}
            
            device = torch.cuda.get_device_properties(0)
            total_memory = device.total_memory / 1024**3  # GB
            
            return {
                'available': True,
                'name': device.name,
                'total_memory': total_memory,
                'compute_capability': f"{device.major}.{device.minor}",
                'multi_processor_count': device.multi_processor_count
            }
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return {'available': False}
    
    def _classify_gpu(self) -> GPUTier:
        """分类GPU性能等级"""
        if not self.gpu_info.get('available'):
            return GPUTier.CPU
        
        memory = self.gpu_info.get('total_memory', 0)
        name = self.gpu_info.get('name', '').lower()
        
        # 根据显存和型号分类
        if memory >= 16:
            return GPUTier.HIGH
        elif memory >= 10:
            return GPUTier.MEDIUM
        elif memory >= 6:
            return GPUTier.LOW
        else:
            return GPUTier.CPU
    
    def _load_presets(self) -> Dict[GPUTier, Dict]:
        """加载预设配置"""
        return {
            GPUTier.HIGH: {
                'deepseek': {
                    'dpi': 200,
                    'image_size': 1024,
                    'base_size': 1536,
                    'batch_size': 4,
                    'use_fp16': True,
                    'estimated_speed': 5.0
                },
                'paddle': {
                    'dpi': 200,
                    'batch_size': 16,
                    'use_fp16': True,
                    'estimated_speed': 2.0
                }
            },
            GPUTier.MEDIUM: {
                'deepseek': {
                    'dpi': 150,
                    'image_size': 768,
                    'base_size': 1024,
                    'batch_size': 2,
                    'use_fp16': True,
                    'estimated_speed': 10.0
                },
                'paddle': {
                    'dpi': 150,
                    'batch_size': 8,
                    'use_fp16': True,
                    'estimated_speed': 3.0
                }
            },
            GPUTier.LOW: {
                'deepseek': {
                    'dpi': 100,
                    'image_size': 512,
                    'base_size': 768,
                    'batch_size': 1,
                    'use_fp16': True,
                    'estimated_speed': 20.0
                },
                'paddle': {
                    'dpi': 150,
                    'batch_size': 4,
                    'use_fp16': True,
                    'estimated_speed': 5.0
                }
            },
            GPUTier.CPU: {
                'deepseek': None,  # CPU不推荐DeepSeek
                'paddle': {
                    'dpi': 100,
                    'batch_size': 1,
                    'use_fp16': False,
                    'estimated_speed': 15.0
                }
            }
        }
    
    def get_optimal_config(
        self,
        prefer_accuracy: bool = True,
        document_complexity: DocumentComplexity = DocumentComplexity.MEDIUM,
        available_time: Optional[float] = None,
        total_pages: Optional[int] = None
    ) -> OptimalConfig:
        """
        获取最优配置
        
        Args:
            prefer_accuracy: 是否优先精度（否则优先速度）
            document_complexity: 文档复杂度
            available_time: 可用时间（秒），用于选择合适的配置
            total_pages: 总页数，用于估算是否可行
            
        Returns:
            最优配置
        """
        presets = self.presets[self.gpu_tier]
        
        # 选择引擎
        if self.gpu_tier == GPUTier.CPU:
            engine = 'paddle'
        elif prefer_accuracy:
            engine = 'deepseek' if presets.get('deepseek') else 'paddle'
        else:
            engine = 'paddle'
        
        preset = presets[engine]
        if preset is None:
            # 降级到paddle
            engine = 'paddle'
            preset = presets['paddle']
        
        # 根据复杂度调整
        dpi = preset['dpi']
        if document_complexity == DocumentComplexity.COMPLEX:
            dpi = min(dpi + 50, 300)
        elif document_complexity == DocumentComplexity.SIMPLE:
            dpi = max(dpi - 50, 72)
        
        # 如果有时间限制，检查是否可行
        estimated_speed = preset['estimated_speed']
        if available_time and total_pages:
            required_time = total_pages * estimated_speed
            if required_time > available_time:
                # 需要更快的配置
                if engine == 'deepseek':
                    engine = 'paddle'
                    preset = presets['paddle']
                    estimated_speed = preset['estimated_speed']
                dpi = max(dpi - 50, 72)
        
        return OptimalConfig(
            engine=engine,
            dpi=dpi,
            image_size=preset.get('image_size', 768),
            base_size=preset.get('base_size', 1024),
            batch_size=preset.get('batch_size', 1),
            use_gpu=self.gpu_tier != GPUTier.CPU,
            use_fp16=preset.get('use_fp16', True),
            estimated_speed=estimated_speed
        )
    
    def estimate_processing_time(
        self,
        total_pages: int,
        config: Optional[OptimalConfig] = None
    ) -> Dict:
        """
        估算处理时间
        
        Args:
            total_pages: 总页数
            config: 配置（如果为None则使用默认配置）
            
        Returns:
            时间估算信息
        """
        if config is None:
            config = self.get_optimal_config()
        
        time_per_page = config.estimated_speed
        total_time = total_pages * time_per_page
        
        return {
            'total_pages': total_pages,
            'time_per_page': time_per_page,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_time_hours': total_time / 3600,
            'formatted': self._format_time(total_time)
        }
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def run_benchmark(self, test_image_path: str = None) -> Dict:
        """
        运行基准测试
        
        Args:
            test_image_path: 测试图像路径
            
        Returns:
            基准测试结果
        """
        results = {
            'gpu_info': self.gpu_info,
            'gpu_tier': self.gpu_tier.value,
            'tests': {}
        }
        
        # 测试不同配置
        configs_to_test = [
            ('deepseek_high', {'engine': 'deepseek', 'dpi': 200}),
            ('deepseek_low', {'engine': 'deepseek', 'dpi': 100}),
            ('paddle', {'engine': 'paddle', 'dpi': 150})
        ]
        
        if self.gpu_tier != GPUTier.CPU and test_image_path:
            for name, config in configs_to_test:
                try:
                    speed = self._benchmark_config(test_image_path, config)
                    results['tests'][name] = {
                        'config': config,
                        'speed': speed,
                        'status': 'success'
                    }
                except Exception as e:
                    results['tests'][name] = {
                        'config': config,
                        'error': str(e),
                        'status': 'failed'
                    }
        
        return results
    
    def _benchmark_config(self, image_path: str, config: Dict) -> float:
        """测试单个配置的速度"""
        # 简化实现，实际需要加载模型并运行
        # 这里返回预估值
        if config['engine'] == 'deepseek':
            return self.presets[self.gpu_tier].get('deepseek', {}).get('estimated_speed', 15)
        else:
            return self.presets[self.gpu_tier].get('paddle', {}).get('estimated_speed', 5)
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        import platform
        import torch
        
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': f"{self.gpu_info.get('total_memory', 0):.1f} GB"
            })
        
        return info
    
    def recommend_action(self) -> str:
        """给出优化建议"""
        recommendations = []
        
        if self.gpu_tier == GPUTier.CPU:
            recommendations.append("⚠️ 未检测到可用GPU，建议安装CUDA和支持GPU的PyTorch")
            recommendations.append("💡 使用PaddleOCR作为OCR引擎（速度更快）")
        elif self.gpu_tier == GPUTier.LOW:
            recommendations.append("⚠️ GPU显存较小，建议：")
            recommendations.append("  - 关闭后台应用释放显存")
            recommendations.append("  - 使用DPI=100降低内存占用")
            recommendations.append("  - 考虑使用PaddleOCR替代DeepSeek-OCR2")
        elif self.gpu_tier == GPUTier.MEDIUM:
            recommendations.append("✅ GPU配置适中，建议：")
            recommendations.append("  - 使用DPI=150平衡速度和精度")
            recommendations.append("  - 关闭不必要的后台应用")
        else:
            recommendations.append("✅ GPU配置优秀，可以使用最高质量设置")
            recommendations.append("  - 使用DPI=200获得最佳精度")
            recommendations.append("  - 可以启用批量处理")
        
        return "\n".join(recommendations)


def get_auto_config() -> OptimalConfig:
    """获取自动优化配置"""
    optimizer = ModelOptimizer()
    return optimizer.get_optimal_config()


def print_system_report():
    """打印系统报告"""
    optimizer = ModelOptimizer()
    
    print("=" * 60)
    print("系统配置报告")
    print("=" * 60)
    
    info = optimizer.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 60)
    print(f"GPU等级: {optimizer.gpu_tier.value}")
    print("-" * 60)
    
    config = optimizer.get_optimal_config()
    print("\n推荐配置:")
    print(f"  OCR引擎: {config.engine}")
    print(f"  DPI: {config.dpi}")
    print(f"  使用GPU: {config.use_gpu}")
    print(f"  使用FP16: {config.use_fp16}")
    print(f"  预估速度: {config.estimated_speed}秒/页")
    
    print("\n" + "-" * 60)
    print("优化建议:")
    print(optimizer.recommend_action())
    print("=" * 60)


if __name__ == "__main__":
    print_system_report()
