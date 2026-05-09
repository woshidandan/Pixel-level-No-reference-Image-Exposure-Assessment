# 图像分析工作流（Grounded-SAM + 曝光评估）

基于 Llama Stack 构建的多轮智能体流水线，用于分割图像中的目标对象并评估其曝光质量。

## 流程概览

```
输入图像
    │
    ▼
[步骤 1] 分割 Agent → 识别需要分割的目标对象
    │
    ▼
[步骤 2] 执行 Agent → 调用 Grounded-SAM 工具，输出带掩码的图像
    │
    ▼
[步骤 3] 曝光 Agent → 对每个分割对象评估曝光分数
    │
    ▼
[步骤 4] 分析 Agent → 生成综合报告（Llama Stack 或 OpenAI）
```

## 环境配置

### 1. 安装依赖

以下两种方式**任选其一**。

#### 方式 A — 复现现有环境（推荐，快速上手）

仓库中附带了通过 `uv export` 导出的 `requirements.txt`，包含当前已验证通过的全部依赖及版本。直接安装即可：

```bash
pip install -r requirements.txt
pip install torch torchvision opencv-python

# 安装本地包
pip install -e Grounded-Segment-Anything/GroundingDINO/
pip install -e Grounded-Segment-Anything/segment_anything/
```

#### 方式 B — 按照官方指南从零安装

如果你更希望从一个干净的环境开始，请参考 [Llama Stack 官方安装指南](https://github.com/llamastack/llama-stack)：

```bash
# 一键安装
curl -LsSf https://github.com/llamastack/llama-stack/raw/main/scripts/install.sh | bash

# 或通过 uv 安装
uv pip install llama-stack

pip install torch torchvision opencv-python

# 安装本地包
pip install -e Grounded-Segment-Anything/GroundingDINO/
pip install -e Grounded-Segment-Anything/segment_anything/
```

### 2. 启动 Llama Stack 服务（另开终端）

流水线连接至 `http://localhost:8321`，使用 `llama4:scout` 模型。

推荐使用 `uv` + `ollama` 方式启动（无需 llama-stack 源码）：

```bash
# 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 ollama 并拉取模型（如尚未完成）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama4:scout

# 启动 Llama Stack 服务
INFERENCE_MODEL=llama4:scout uv run --with llama-stack llama stack build --template ollama --image-type venv --run
```

> `llama-stack` 包由 `uv` 自动从 PyPI 下载，无需本地 llama-stack 源文件。

### 3.（可选）配置 OpenAI 环境变量

仅在最终分析步骤使用 `use_openai=True` 时需要。

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://..."   # 可选，用于自定义端点
```

## 使用方法

### 快速开始

运行前，请先打开 `Grounding_SAM_Exposure_workflow/image_analysis_workflow.py`，编辑文件底部的 `main()` 函数，设置**你自己的图片路径和查询文本**：

```python
def main():
    workflow = ImageAnalysisWorkflow()

    # ↓↓↓ 自定义以下两个变量 ↓↓↓
    query = "请分析该图像中人物的曝光水平"          # 你的分析请求
    image_path = "/path/to/your/image.jpg"         # 你的输入图片路径
    # ↑↑↑ 自定义以上两个变量 ↑↑↑

    results = workflow.analyze_image(query, image_path)
    ...
```

- **`image_path`** — 待分析图片的绝对路径或相对路径（例如 `"test_images/photo.jpg"`）。
- **`query`** — 自然语言指令，告诉流水线*需要分割和评估哪些目标*。不同的查询会产生不同的分割结果，例如：
  - `"分析人物的曝光"` → 分割 `person`
  - `"评估汽车和建筑的曝光"` → 分割 `car, building`
  - `"检查天空和前景的曝光质量"` → 分割 `sky, foreground`

然后**从项目根目录运行**（不要在工作流文件夹内运行）：

```bash
python Grounding_SAM_Exposure_workflow/image_analysis_workflow.py
```

### 编程方式调用

以编程方式调用工作流：

```python
from Grounding_SAM_Exposure_workflow.image_analysis_workflow import ImageAnalysisWorkflow

workflow = ImageAnalysisWorkflow(
    base_url="http://localhost:8321",
    openai_api_key="...",        # 可选
    openai_model="gpt-4o"        # 可选
)

results = workflow.analyze_image(
    user_query="请分析该图像中人物的曝光水平",
    image_path="/path/to/your/image.jpg",
    use_openai=False             # 设为 True 则使用 OpenAI 进行最终分析
)

print(workflow.generate_comprehensive_report(results))
```

## 输出结果

- **分割图像**：保存至 `grounded_sam_outputs/`
- **曝光热力图**：保存至 `exposure_heatmaps/`
- **分析报告**：以 Markdown 格式打印至标准输出
