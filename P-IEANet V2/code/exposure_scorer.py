import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np # Added for background mask

# 添加P-IEANet目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
p_ieanet_dir = os.path.join(current_dir, 'P-IEANet')
sys.path.append(p_ieanet_dir)

from wavelet_network import Wavelet_Net

class ExposureScorer:
    """曝光评估器，使用Wavelet_Net模型评估图像的曝光水平"""
    
    def __init__(self, model_path=None):
        """
        初始化曝光评估器
        
        Args:
            model_path (str): 模型权重文件路径，如果为None则使用默认路径
        """
        if model_path is None:
            model_path = os.path.join(p_ieanet_dir, "wavelet_epoch_2_val_loss0.03022034629540784.pth")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        
        # 图像预处理参数
        self.IMAGE_NET_MEAN = [0., 0., 0.]
        self.IMAGE_NET_STD = [1., 1., 1.]
        self.normalize = transforms.Normalize(
            mean=self.IMAGE_NET_MEAN,
            std=self.IMAGE_NET_STD
        )
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            self.normalize
        ])
    
    def _load_model(self, model_path):
        """加载Wavelet_Net模型"""
        model = Wavelet_Net()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def read_image(self, image_path):
        """读取图像并转换为tensor"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        return image_tensor
    
    def evaluate_image(self, image_path, ignore_background=True):
        """
        评估图像的曝光水平
        
        Args:
            image_path (str): 图像文件路径
            ignore_background (bool): 是否忽略背景区域（透明或黑色）
            
        Returns:
            tuple: (exposure_tensor, average_score, background_mask)
                - exposure_tensor: 曝光评估tensor，每个像素点代表该位置的曝光评分
                - average_score: 平均曝光评分（忽略背景区域）
                - background_mask: 背景掩码（当ignore_background=True时）
        """
        try:
            # 读取图像
            image = self.read_image(image_path)
            height, width = image.shape[2], image.shape[3]
            origin_trans = transforms.Resize((height, width))
            
            # 预处理图像
            image = self.transform(image)
            image = torch.cat([image, image], dim=0)  # 双输入，模拟make_heatmap.py
            image = image.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                exposure = self.model(image)
                exposure = origin_trans(exposure[0])[0]  # 取第一个输出并调整尺寸
                exposure = exposure * -1  # 翻转符号，使高分代表好曝光
            
            # 如果需要忽略背景，创建背景掩码
            background_mask = None
            if ignore_background:
                # 读取原始图像来检测背景
                original_pil = Image.open(image_path)
                
                if original_pil.mode == 'RGBA':
                    # 对于RGBA图像，检测透明区域
                    original_array = np.array(original_pil)
                    alpha_channel = original_array[:, :, 3]
                    background_mask = alpha_channel > 0  # 非透明区域
                else:
                    # 对于RGB图像，检测黑色区域（可能是背景）
                    original_array = np.array(original_pil)
                    # 检测接近黑色的像素（RGB值都很小）
                    black_threshold = 30  # 可以调整这个阈值
                    background_mask = np.any(original_array > black_threshold, axis=2)
                
                # 将背景掩码转换为tensor
                background_mask_tensor = torch.from_numpy(background_mask).float().to(self.device)
                
                # 只计算非背景区域的平均分数
                valid_exposure = exposure * background_mask_tensor
                valid_pixels = background_mask_tensor.sum()
                
                if valid_pixels > 0:
                    average_score = valid_exposure.sum() / valid_pixels
                else:
                    average_score = exposure.mean().item()
            else:
                # 原来的方法：计算所有像素的平均分数
                average_score = exposure.mean().item()
            
            return exposure, average_score, background_mask
            
        except Exception as e:
            raise Exception(f"曝光评估失败: {str(e)}")
    
    def generate_heatmap(self, exposure_tensor, output_path, background_mask=None):
        """
        生成曝光热力图
        
        Args:
            exposure_tensor: 曝光评估tensor
            output_path (str): 热力图保存路径
            background_mask: 背景掩码，用于标识透明区域
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # 归一化tensor到[0,1]范围
        tensor = (exposure_tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        tensor_np = tensor.cpu().numpy()
        
        # 创建蓝-白-红颜色映射
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝色-白色-红色
        cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
        
        # 创建热力图
        plt.figure(figsize=(6, 6))
        
        if background_mask is not None:
            # 如果有背景掩码，先显示普通热力图
            im = plt.imshow(tensor_np, cmap=cmap)
            
            # 将背景掩码转换为numpy数组
            if isinstance(background_mask, torch.Tensor):
                background_mask_np = background_mask.cpu().numpy()
            else:
                background_mask_np = background_mask
            
            # 创建透明掩码：非背景区域设为1（不透明），背景区域设为0（透明）
            alpha_mask = background_mask_np.astype(float)
            
            # 设置图像的alpha通道
            im.set_alpha(alpha_mask)
        else:
            # 原来的方法：显示普通热力图
            plt.imshow(tensor_np, cmap=cmap)
        
        plt.axis("off")
        
        # 保存热力图，支持透明度
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close() 