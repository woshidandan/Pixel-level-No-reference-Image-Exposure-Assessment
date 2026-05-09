from llama_stack_client.types.tool_def_param import Parameter
from llama_stack_client.lib.agents.client_tool import ClientTool
from grounded_sam_segmenter import GroundedSamSegmenter
import os
import cv2
import numpy as np
from PIL import Image
import uuid
import torch

class GroundedSamTool(ClientTool):
    def __init__(self):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 添加防重复执行机制
        self._execution_cache = {}
        
        # 模型路径配置
        grounding_dino_config_path = os.path.join(
            current_dir, 
            'Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        )
        grounding_dino_checkpoint_path = os.path.join(
            current_dir, 
            'Grounded-Segment-Anything/groundingdino_swint_ogc.pth'
        )
        sam_checkpoint_path = os.path.join(
            current_dir, 
            'Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        )
        
        # 检查模型文件是否存在
        self._check_model_files(
            grounding_dino_config_path,
            grounding_dino_checkpoint_path,
            sam_checkpoint_path
        )
        
        # 初始化分割器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.segmenter = GroundedSamSegmenter(
            grounding_dino_config_path=grounding_dino_config_path,
            grounding_dino_checkpoint_path=grounding_dino_checkpoint_path,
            sam_checkpoint_path=sam_checkpoint_path,
            sam_model_type="vit_h",
            device=device
        )
        
        # 创建输出目录
        self.output_dir = os.path.join(current_dir, 'grounded_sam_outputs')
        os.makedirs(self.output_dir, exist_ok=True)

    def _check_model_files(self, config_path, dino_checkpoint, sam_checkpoint):
        """检查模型文件是否存在"""
        missing_files = []
        if not os.path.exists(config_path):
            missing_files.append(f"GroundingDINO config: {config_path}")
        if not os.path.exists(dino_checkpoint):
            missing_files.append(f"GroundingDINO checkpoint: {dino_checkpoint}")
        if not os.path.exists(sam_checkpoint):
            missing_files.append(f"SAM checkpoint: {sam_checkpoint}")
            
        if missing_files:
            raise FileNotFoundError(
                f"Missing model files:\n" + "\n".join(missing_files) + 
                "\n\nPlease download the required model files and place them in the correct locations."
            )

    def get_name(self):
        return "grounded_sam_segment"

    def get_description(self):
        return "Segment specific objects in an image using Grounded-SAM model. Takes a text prompt describing what objects to segment and returns the segmented images."

    def get_params_definition(self):
        return {
            "image_path": Parameter(
                name="image_path",
                description="Local file path to the image",
                parameter_type="str",
                required=True
            ),
            "text_prompt": Parameter(
                name="text_prompt",
                description="Text description of objects to segment (e.g., 'the dog', 'a car', 'person')",
                parameter_type="str",
                required=True
            ),
            "box_threshold": Parameter(
                name="box_threshold",
                description="Confidence threshold for object detection (0.0-1.0, default: 0.3)",
                parameter_type="float",
                required=False
            ),
            "text_threshold": Parameter(
                name="text_threshold",
                description="Text confidence threshold (0.0-1.0, default: 0.25)",
                parameter_type="float",
                required=False
            )
        }

    def apply_mask_to_image(self, image, mask, output_path, use_transparent_background=True):
        """
        将mask应用到原图上，生成分割后的局部图片
        
        Args:
            image: 原始图像数组
            mask: 分割掩码
            output_path: 输出路径
            use_transparent_background: 是否使用透明背景（默认True）
        """
        # 确保mask是正确的形状
        if len(mask.shape) == 4:  # (1, 1, H, W) -> (H, W)
            mask = mask.squeeze()
        elif len(mask.shape) == 3:  # (1, H, W) -> (H, W)
            mask = mask.squeeze(0)
        
        # 确保mask是布尔类型
        mask = mask.astype(bool)
        
        # Debug information removed to reduce noise
        
        if use_transparent_background:
            # 创建RGBA图像，背景透明
            if image.shape[2] == 3:  # RGB图像
                # 创建RGBA图像
                rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba_image[:, :, :3] = image  # 复制RGB通道
                rgba_image[:, :, 3] = 255  # 设置alpha通道为255（不透明）
                
                # 将非mask区域的alpha设为0（透明）
                rgba_image[~mask, 3] = 0
                
                # 转换为PIL图像并保存为PNG（支持透明）
                pil_image = Image.fromarray(rgba_image, 'RGBA')
                pil_image.save(output_path, 'PNG')
            else:
                # 如果已经是RGBA，直接处理
                rgba_image = image.copy()
                rgba_image[~mask, 3] = 0
                pil_image = Image.fromarray(rgba_image, 'RGBA')
                pil_image.save(output_path, 'PNG')
        else:
            # 原来的方法：使用黑色背景
            segmented_image = np.zeros_like(image)
            mask_3d = mask[:, :, np.newaxis]  # (H, W, 1)
            segmented_image = np.where(mask_3d, image, segmented_image)
            pil_image = Image.fromarray(segmented_image)
            pil_image.save(output_path, 'PNG')
        
        return output_path

    def run_impl(self, image_path: str, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        try:
            print(f"🔍 GroundedSamTool.run_impl called with: {image_path}, {text_prompt}")
            
            # 创建缓存键
            cache_key = f"{image_path}_{text_prompt}_{box_threshold}_{text_threshold}"
            
            # 检查是否已经执行过相同的请求
            if cache_key in self._execution_cache:
                print(f"🔄 Using cached result for: {cache_key}")
                return self._execution_cache[cache_key]
            
            # 检查输入图片是否存在
            if not os.path.exists(image_path):
                return {
                    "error": f"Image file not found: {image_path}",
                    "source": "GroundedSamTool"
                }
            
            # 生成唯一的会话ID用于文件命名
            session_id = str(uuid.uuid4())[:8]
            
            # 读取原图
            original_image = np.array(Image.open(image_path))
            
            # 获取分割结果
            masks, scores, boxes, phrases = self.segmenter.segment(
                image_path, 
                text_prompt, 
                box_threshold, 
                text_threshold
            )
            
            if len(masks) == 0:
                return {
                    "error": f"No objects found matching the text prompt: '{text_prompt}'",
                    "source": "GroundedSamTool",
                    "text_prompt": text_prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold
                }
            
            # 处理每个mask
            segmented_images = []
            for i in range(len(masks)):
                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = f"{base_name}_grounded_{i:03d}_{session_id}.png"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # 应用mask并保存图片
                self.apply_mask_to_image(original_image, masks[i], output_path)
                
                # 记录结果
                segmented_images.append({
                    "mask_id": i,
                    "image_path": output_path,
                    "score": float(scores[i]),
                    "filename": output_filename,
                    "phrase": phrases[i] if i < len(phrases) else f"object_{i}",
                    "box": boxes[i].tolist() if i < len(boxes) else None
                })
            
            result = {
                "segmented_images": segmented_images,
                "total_masks": len(masks),
                "output_directory": self.output_dir,
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "source": "GroundedSamTool"
            }
            
            # 缓存结果
            self._execution_cache[cache_key] = result
            print(f"💾 Cached result for: {cache_key}")
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "source": "GroundedSamTool"
            }

    async def async_run_impl(self, image_path: str, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        return self.run_impl(image_path, text_prompt, box_threshold, text_threshold) 