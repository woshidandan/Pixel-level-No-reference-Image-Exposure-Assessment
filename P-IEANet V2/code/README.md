# Image Analysis Workflow (Grounded-SAM + Exposure Evaluation)

A multi-turn agent pipeline that segments objects in images and evaluates their exposure quality, powered by Llama Stack.

## Pipeline Overview

```
Input Image
    │
    ▼
[Step 1] Segmentation Agent → identifies objects to segment
    │
    ▼
[Step 2] Executor Agent → calls Grounded-SAM tool, outputs masked images
    │
    ▼
[Step 3] Exposure Agent → evaluates exposure score for each segmented object
    │
    ▼
[Step 4] Analysis Agent → generates comprehensive report (Llama Stack or OpenAI)
```

## Setup

### 1. Install dependencies

Choose **one** of the two methods below.

#### Option A — Reproduce the existing environment (recommended for quick start)

The repository ships a `requirements.txt` exported from the tested working environment via `uv export`. You can install all pinned dependencies directly:

```bash
pip install -r requirements.txt
pip install torch torchvision opencv-python

# Install local packages
pip install -e Grounded-Segment-Anything/GroundingDINO/
pip install -e Grounded-Segment-Anything/segment_anything/
```

#### Option B — Install from scratch following the official guide

If you prefer to set up a fresh environment, follow the [official Llama Stack installation guide](https://github.com/llamastack/llama-stack):

```bash
# One-line install
curl -LsSf https://github.com/llamastack/llama-stack/raw/main/scripts/install.sh | bash

# Or install via uv
uv pip install llama-stack

pip install torch torchvision opencv-python

# Install local packages
pip install -e Grounded-Segment-Anything/GroundingDINO/
pip install -e Grounded-Segment-Anything/segment_anything/
```

### 2. Start Llama Stack server (separate terminal)

The pipeline connects to `http://localhost:8321` and uses the `llama4:scout` model.

The recommended way is to use `uv` + `ollama` (no llama-stack source code needed):

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ollama and pull the model (if not already done)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama4:scout

# Start the Llama Stack server
INFERENCE_MODEL=llama4:scout uv run --with llama-stack llama stack build --template ollama --image-type venv --run
```

> The `llama-stack` package is downloaded automatically by `uv` from PyPI. No local llama-stack source files are required.

### 3. (Optional) Set OpenAI environment variables

Required only if using `use_openai=True` for the final analysis step.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://..."   # optional, for custom endpoint
```

## Usage

### Quick start

Before running, open `Grounding_SAM_Exposure_workflow/image_analysis_workflow.py` and edit the `main()` function at the bottom of the file to set **your own image path and query**:

```python
def main():
    workflow = ImageAnalysisWorkflow()

    # ↓↓↓ Customize these two variables ↓↓↓
    query = "Please analyze the exposure level of this image"   # your analysis request
    image_path = "/path/to/your/image.jpg"                                    # your input image
    # ↑↑↑ Customize these two variables ↑↑↑

    results = workflow.analyze_image(query, image_path)
    ...
```

- **`image_path`** — absolute or relative path to the image you want to analyze (e.g. `"test_images/photo.jpg"`).
- **`query`** — a natural-language instruction that tells the pipeline *what* to segment and evaluate. Different queries lead to different segmentation targets. For example:
  - `"Analyze the exposure of the person"` → segments `person`
  - `"Evaluate the exposure of the car and the building"` → segments `car, building`
  - `"Check the exposure quality of the sky and the foreground"` → segments `sky, foreground`

Then run from the **project root** (not from inside the workflow folder):

```bash
python Grounding_SAM_Exposure_workflow/image_analysis_workflow.py
```

### Programmatic usage

To use the workflow programmatically:

```python
from Grounding_SAM_Exposure_workflow.image_analysis_workflow import ImageAnalysisWorkflow

workflow = ImageAnalysisWorkflow(
    base_url="http://localhost:8321",
    openai_api_key="...",        # optional
    openai_model="gpt-4o"        # optional
)

results = workflow.analyze_image(
    user_query="Please analyze the exposure level of the person in this image",
    image_path="/path/to/your/image.jpg",
    use_openai=False             # set True to use OpenAI for final analysis
)

print(workflow.generate_comprehensive_report(results))
```

## Output

- **Segmented images**: saved to `grounded_sam_outputs/`
- **Exposure heatmaps**: saved to `exposure_heatmaps/`
- **Report**: printed to stdout (Markdown format)

## Rule RAG and Dual-Grounding

The workflow now adds an intermediate evidence layer before the final report:

- **Rule RAG** retrieves exposure-assessment rules from `rule_rag.py` using the user query, object labels, and exposure statistics.
- **Dual-Grounding** writes a JSON manifest that aligns visual evidence and textual evidence:
  - visual evidence: region id, object label, bounding box, segmented mask path, heatmap path
  - textual evidence: retrieved rule ids, exposure scores, and grounded reasoning statements

The manifest is saved beside the segmented outputs as:

```bash
grounded_sam_outputs/<image_name>_dual_grounding.json
```

You can provide a custom JSON rule library:

```python
workflow = ImageAnalysisWorkflow(rules_path="my_exposure_rules.json")
```

## Trainable Reasoning-Grounding Model

`train_reasoning_grounding.py` provides a lightweight trainable approximation of the paper pipeline. It uses a Quality Encoder, Visual Encoder, and Question Encoder to produce tokens, fuses them with a compact Transformer LLM Backbone, and jointly learns:

- a segmentation mask output
- an exposure/reasoning text output

Training data is JSONL, one sample per line:

```json
{"image": "images/a.jpg", "mask": "masks/a.png", "question": "Assess the exposure of the main person.", "text": "The person is underexposed in the face region. Increase local exposure and protect highlights."}
```

The training script can also use Rule RAG. Add any of these optional fields:

- `rule_context`: preformatted rule evidence text
- `rules`: retrieved rule dictionaries
- `grounding_records`: exposure/region records used to retrieve rules automatically

Example with automatic Rule RAG:

```json
{"image": "images/a.jpg", "mask": "masks/a.png", "question": "Assess the exposure of the main person.", "grounding_records": [{"label": "person", "average_exposure_score": -0.31, "exposure_std": 0.37}], "text": "The person is underexposed and unevenly lit. Increase local exposure on the subject."}
```

Train:

```bash
python train_reasoning_grounding.py train --train-jsonl train.jsonl --image-root /path/to/data --rules-path my_exposure_rules.json --output-dir runs/iea_reasoning
```

Train directly with the provided `Dataset Example` layout:

```bash
python train_reasoning_grounding.py train --dataset-format dataset_example --dataset-root "Dataset Example" --image-root /path/to/source_images --output-dir runs/iea_reasoning
```

For this layout, the script reads:

```text
Dataset Example/
  analysis_only/<sample_id>.json
  segmentations/<sample_id>/<mask_pngs>
```

The source image is expected under `--image-root` with the same stem as the JSON file, for example `<sample_id>.jpg`. If the source image is not present, the script can still build a pseudo input from the segmented PNGs, which is useful for checking the GT format, but real training should use the normal original images.

Infer:

```bash
python train_reasoning_grounding.py infer --checkpoint runs/iea_reasoning/reasoning_grounding_model.pt --image /path/to/image.jpg --question "Assess the exposure of the main person." --rules-path my_exposure_rules.json --grounding-records-json "[{\"label\":\"person\",\"average_exposure_score\":-0.31,\"exposure_std\":0.37}]" --output-image outputs/segmented.png --output-text outputs/report.txt
```
