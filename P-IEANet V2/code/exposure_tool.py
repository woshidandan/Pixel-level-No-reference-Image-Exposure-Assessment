from typing import Dict, Any
from llama_stack_client.types.tool_def_param import Parameter
from llama_stack_client.lib.agents.client_tool import ClientTool
from exposure_scorer import ExposureScorer
import os
import base64

class ExposureTool(ClientTool):
    """A tool that provides exposure evaluation for images."""

    def __init__(self, output_dir=None):
        super().__init__()
        # 初始化曝光评估器
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if output_dir is None:
            output_dir = os.path.join(current_dir, 'exposure_heatmaps')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.scorer = ExposureScorer()

    def get_name(self) -> str:
        return "evaluate_exposure"

    def get_description(self) -> str:
        return "Evaluate the exposure level of an image. This tool provides pixel-level exposure scores and generates a heatmap visualization. Higher scores indicate better exposure."

    def get_params_definition(self) -> Dict[str, Parameter]:
        return {
            "image_path": Parameter(
                name="image_path",
                description="Local file path to the image",
                parameter_type="str",
                required=True
            ),
            "ignore_background": Parameter(
                name="ignore_background",
                description="Whether to ignore background regions (transparent or black) when calculating exposure scores",
                parameter_type="bool",
                required=False
            )
        }

    def run_impl(self, image_path: str, ignore_background: bool = True) -> Any:
        try:
            # 评估图像曝光
            exposure_tensor, average_score, background_mask = self.scorer.evaluate_image(image_path, ignore_background)
            
            # 生成热力图文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            heatmap_filename = f"{base_name}_exposure_heatmap.png"
            heatmap_path = os.path.join(self.output_dir, heatmap_filename)
            
            # 生成热力图，传递背景掩码
            self.scorer.generate_heatmap(exposure_tensor, heatmap_path, background_mask)
            
            # 计算统计信息
            min_score = exposure_tensor.min().item()
            max_score = exposure_tensor.max().item()
            std_score = exposure_tensor.std().item()
            
            # 移除大的base64数据，只返回关键评分信息
            return {
                "average_exposure_score": round(float(average_score), 4),
                "min_exposure_score": round(float(min_score), 4),
                "max_exposure_score": round(float(max_score), 4),
                "exposure_std": round(float(std_score), 4),
                "heatmap_path": heatmap_path,
                "heatmap_filename": heatmap_filename,
                "ignore_background": ignore_background,
                "has_background_mask": background_mask is not None,
                "source": "ExposureTool",
                "description": f"Exposure evaluation completed. Average score: {round(float(average_score), 4)}. Heatmap saved to: {heatmap_path}"
            }
        except Exception as e:
            return {
                "error": str(e),
                "source": "ExposureTool"
            }

    async def async_run_impl(self, image_path: str, ignore_background: bool = True) -> Any:
        return self.run_impl(image_path, ignore_background) 