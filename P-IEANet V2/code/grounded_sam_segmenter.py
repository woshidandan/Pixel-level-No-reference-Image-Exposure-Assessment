import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# 添加GroundingDINO和segment_anything到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
grounding_dino_path = os.path.join(current_dir, 'Grounded-Segment-Anything/GroundingDINO')
segment_anything_path = os.path.join(current_dir, 'Grounded-Segment-Anything/segment_anything')

# 直接添加路径到 sys.path
sys.path.insert(0, grounding_dino_path)
sys.path.insert(0, segment_anything_path)

# Grounding DINO imports (不使用 GroundingDINO 前缀，因为已经添加了路径)
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor

class GroundedSamSegmenter:
    def __init__(self, 
                 grounding_dino_config_path,
                 grounding_dino_checkpoint_path,
                 sam_checkpoint_path,
                 sam_model_type="vit_h",
                 device="cpu"):
        """
        初始化Grounded-SAM分割器
        
        Args:
            grounding_dino_config_path: GroundingDINO配置文件路径
            grounding_dino_checkpoint_path: GroundingDINO模型权重路径
            sam_checkpoint_path: SAM模型权重路径
            sam_model_type: SAM模型类型 (vit_b, vit_l, vit_h)
            device: 运行设备
        """
        self.device = device
        
        # 初始化GroundingDINO模型
        self.grounding_dino_model = self._load_grounding_dino_model(
            grounding_dino_config_path, 
            grounding_dino_checkpoint_path
        )
        
        # 初始化SAM模型
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam_model.to(device=device)
        self.sam_predictor = SamPredictor(self.sam_model)
        
        # 确保SAM模型在正确的设备上
        print(f"Grounded-SAM initialized on device: {self.device}")
        
    def _load_grounding_dino_model(self, config_path, checkpoint_path):
        """加载GroundingDINO模型"""
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"GroundingDINO model loaded: {load_res}")
        model = model.to(self.device)  # 确保模型移动到正确的设备
        model.eval()
        return model
    
    def _load_image(self, image_path):
        """加载和预处理图像"""
        image_pil = Image.open(image_path).convert("RGB")
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image
    
    def _get_grounding_output(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """使用GroundingDINO进行目标检测 - 参考官方inference.py的实现"""
        # 预处理文本提示
        text_prompt = text_prompt.lower().strip()
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."
            
        # 确保模型在正确的设备上
        self.grounding_dino_model = self.grounding_dino_model.to(self.device)
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_dino_model(image[None], captions=[text_prompt])
            
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # 添加详细的调试信息
        print(f"Debug - prediction_logits shape: {prediction_logits.shape}")
        print(f"Debug - prediction_boxes shape: {prediction_boxes.shape}")
        
        # 计算每个query的最大置信度
        max_scores = prediction_logits.max(dim=1)[0]
        print(f"Debug - max_scores shape: {max_scores.shape}")
        print(f"Debug - max_scores type: {type(max_scores)}")
        print(f"Debug - max_scores device: {max_scores.device}")
        
        # 创建过滤掩码
        mask = max_scores > box_threshold
        print(f"Debug - mask shape: {mask.shape}")
        print(f"Debug - mask type: {type(mask)}")
        print(f"Debug - mask device: {mask.device}")
        print(f"Debug - mask sum: {mask.sum()}")
        
        # 检查维度匹配
        if mask.shape[0] != prediction_logits.shape[0]:
            print(f"Error: mask shape {mask.shape} doesn't match prediction_logits shape {prediction_logits.shape}")
            # 尝试修复：确保mask是一维的
            if len(mask.shape) == 0:  # 如果是标量
                mask = mask.unsqueeze(0)
            elif len(mask.shape) > 1:  # 如果是多维的
                mask = mask.squeeze()
            print(f"Debug - fixed mask shape: {mask.shape}")
        
        # 应用过滤
        if mask.sum() > 0:
            logits = prediction_logits[mask]  # logits.shape = (n, 256)
            boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
            print(f"Debug - filtered logits shape: {logits.shape}")
            print(f"Debug - filtered boxes shape: {boxes.shape}")
        else:
            print("Debug - No objects detected above threshold")
            return torch.empty(0, 4), []
        
        # 获取预测短语
        tokenizer = self.grounding_dino_model.tokenizer
        tokenized = tokenizer(text_prompt)
        
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '') + f"({str(logit.max().item())[:4]})"
            for logit in logits
        ]
        
        return boxes, phrases
    
    def segment(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        使用Grounded-SAM分割图像中的指定对象
        
        Args:
            image_path: 图像文件路径
            text_prompt: 文本提示，描述要分割的对象
            box_threshold: 边界框阈值
            text_threshold: 文本阈值
            
        Returns:
            masks: 分割掩码列表
            scores: 置信度分数列表
            boxes: 边界框列表
            phrases: 预测短语列表
        """
        # 加载图像
        image_pil, image = self._load_image(image_path)
        
        # 使用GroundingDINO检测目标
        boxes, phrases = self._get_grounding_output(
            image, text_prompt, box_threshold, text_threshold
        )
        
        if len(boxes) == 0:
            return [], [], [], []
        
        # 准备SAM输入
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_cv)
        
        # 转换边界框坐标 - 参考官方demo的处理方式
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] += boxes[i][:2]
        
        # 确保boxes在CPU上，然后转换到GPU
        boxes = boxes.cpu()
        print(f"Debug - boxes shape before transform: {boxes.shape}")
        print(f"Debug - image_cv shape: {image_cv.shape}")
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes, image_cv.shape[:2]
        ).to(self.device)
        print(f"Debug - transformed_boxes shape: {transformed_boxes.shape}")
        print(f"Debug - transformed_boxes device: {transformed_boxes.device}")
        
        # 使用SAM进行分割 - 确保所有tensor都在正确的设备上
        try:
            # 使用正确的返回值名称：masks, iou_predictions, low_res_masks
            masks, iou_predictions, low_res_masks = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),  # 确保boxes在正确的设备上
                multimask_output=False,
            )
            print(f"Debug - SAM prediction successful")
            print(f"Debug - masks shape: {masks.shape}")
            print(f"Debug - iou_predictions shape: {iou_predictions.shape}")
            
            # 使用iou_predictions作为scores
            scores = iou_predictions.squeeze(0)  # 移除batch维度
            
        except Exception as e:
            print(f"Debug - SAM prediction failed: {e}")
            raise e
        
        return masks.cpu().numpy(), scores.cpu().numpy(), boxes.numpy(), phrases 