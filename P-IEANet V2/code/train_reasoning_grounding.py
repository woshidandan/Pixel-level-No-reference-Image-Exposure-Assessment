import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from rule_rag import RuleRAGRetriever


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class SimpleTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.token_to_id = vocab or {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        tokens: List[str] = []
        current = []
        for char in text:
            if char.isalnum() or char in {"_", "-"}:
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
                if char.strip():
                    tokens.append(char)
        if current:
            tokens.append("".join(current))
        return tokens

    def build(self, texts: List[str], min_freq: int = 1, max_vocab: int = 12000) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for token, count in sorted_items:
            if count < min_freq:
                continue
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
            if len(self.token_to_id) >= max_vocab:
                break
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.bos_id]
        ids.extend(self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text))
        ids.append(self.eos_id)
        ids = ids[:max_len]
        ids.extend([self.pad_id] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if token in {"<pad>", "<bos>"}:
                continue
            if token == "<eos>":
                break
            words.append(token)
        return " ".join(words).replace(" ,", ",").replace(" .", ".").replace(" :", ":")

    def to_dict(self) -> Dict[str, int]:
        return dict(self.token_to_id)


class HFTokenizerWrapper:
    """Adapter that lets HuggingFace tokenizers fit the dataset interface."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, local_files_only: bool = False) -> "HFTokenizerWrapper":
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=True,
                local_files_only=local_files_only,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=False,
                local_files_only=local_files_only,
            )
        return cls(tokenizer)

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_id(self) -> int:
        token_id = self.tokenizer.bos_token_id
        if token_id is None:
            token_id = self.tokenizer.cls_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.unk_id
        return int(token_id)

    @property
    def eos_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.tokenizer.sep_token_id
        if token_id is None:
            token_id = self.tokenizer.bos_token_id
        if token_id is None:
            token_id = self.unk_id
        return int(token_id)

    @property
    def unk_id(self) -> int:
        return int(self.tokenizer.unk_token_id or self.pad_id)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def build(self, texts: List[str], min_freq: int = 1, max_vocab: int = 12000) -> None:
        return None

    def to_dict(self) -> Dict[str, str]:
        return {
            "type": "huggingface",
            "model_name_or_path": self.tokenizer.name_or_path,
        }

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"][0].long()

    def decode(self, ids: List[int]) -> str:
        filtered = []
        for idx in ids:
            idx = int(idx)
            if idx == self.pad_id or idx == self.bos_id:
                continue
            if idx == self.eos_id:
                break
            filtered.append(idx)
        return self.tokenizer.decode(filtered, skip_special_tokens=True).strip()


@dataclass
class SamplePaths:
    image_path: str
    mask_path: Optional[str]
    question: str
    rule_context: str
    text: str
    segmentation_paths: Optional[List[str]] = None


class ReasoningGroundingDataset(Dataset):
    """
    JSONL schema:
    {"image": "images/a.jpg", "mask": "masks/a.png", "question": "Assess the person.", "text": "..."}

    Optional RAG fields:
    - "rule_context": preformatted rule text appended to the question
    - "rules": list of rule dicts with rule_id/title/condition/guidance fields
    - "grounding_records": list of exposure/region records for automatic RuleRAGRetriever lookup

    The mask is optional. If omitted, the segmentation loss uses an empty mask.
    The question is optional. If omitted, a default exposure-assessment prompt is used.
    """

    DEFAULT_QUESTION = "Assess image exposure quality and segment the relevant region."

    def __init__(
        self,
        jsonl_path: Optional[str],
        image_root: str = "",
        tokenizer: Optional[SimpleTokenizer] = None,
        image_size: int = 256,
        max_text_len: int = 128,
        max_question_len: int = 64,
        rules_path: str = "",
        rag_top_k: int = 5,
        dataset_format: str = "auto",
        dataset_root: str = "",
        analysis_dir: str = "analysis_only",
        segmentation_dir: str = "segmentations",
        image_extensions: str = ".jpg,.jpeg,.png,.tif,.tiff,.bmp",
    ):
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.max_question_len = max_question_len
        self.rule_rag = RuleRAGRetriever(rules_path=rules_path or None)
        self.rag_top_k = rag_top_k
        self.dataset_format = dataset_format
        self.dataset_root = dataset_root
        self.analysis_dir = analysis_dir
        self.segmentation_dir = segmentation_dir
        self.image_extensions = [ext.strip() for ext in image_extensions.split(",") if ext.strip()]
        self.samples = self._load_samples(jsonl_path)
        self.tokenizer = tokenizer or SimpleTokenizer()

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_samples(self, jsonl_path: Optional[str]) -> List[SamplePaths]:
        if self.dataset_format not in {"auto", "jsonl", "dataset_example"}:
            raise ValueError("dataset_format must be one of: auto, jsonl, dataset_example")

        if self.dataset_format == "dataset_example":
            return self._load_dataset_example(self.dataset_root or jsonl_path)

        if self.dataset_format == "auto":
            candidate_root = self.dataset_root or jsonl_path
            if candidate_root and os.path.isdir(candidate_root):
                analysis_path = os.path.join(candidate_root, self.analysis_dir)
                segmentation_path = os.path.join(candidate_root, self.segmentation_dir)
                if os.path.isdir(analysis_path) and os.path.isdir(segmentation_path):
                    return self._load_dataset_example(candidate_root)

        if not jsonl_path:
            raise ValueError("A JSONL path is required unless --dataset-format dataset_example is used")

        samples: List[SamplePaths] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text") or item.get("report") or item.get("answer")
                question = item.get("question") or item.get("query") or self.DEFAULT_QUESTION
                if not item.get("image") or not text:
                    raise ValueError(f"Line {line_no} must contain image and text/report/answer")
                rule_context = self._build_rule_context(item, question)
                samples.append(SamplePaths(item["image"], item.get("mask"), question, rule_context, text))
        return samples

    def _load_dataset_example(self, dataset_root: Optional[str]) -> List[SamplePaths]:
        if not dataset_root:
            raise ValueError("--dataset-root or --train-jsonl must point to the Dataset Example directory")

        analysis_root = os.path.join(dataset_root, self.analysis_dir)
        segmentation_root = os.path.join(dataset_root, self.segmentation_dir)
        if not os.path.isdir(analysis_root):
            raise FileNotFoundError(f"analysis directory not found: {analysis_root}")
        if not os.path.isdir(segmentation_root):
            raise FileNotFoundError(f"segmentation directory not found: {segmentation_root}")

        samples: List[SamplePaths] = []
        for filename in sorted(os.listdir(analysis_root)):
            if not filename.lower().endswith(".json"):
                continue
            json_path = os.path.join(analysis_root, filename)
            sample_id = os.path.splitext(filename)[0]
            with open(json_path, "r", encoding="utf-8") as f:
                item = json.load(f)

            global_saliency = item.get("global_saliency_analysis", {})
            global_strategy = item.get("global_exposure_strategy", {})
            image_path = item.get("image") or item.get("image_filename") or sample_id

            for region in item.get("semantic_regions", []):
                mask_filename = region.get("mask_filename")
                mask_path = os.path.abspath(os.path.join(segmentation_root, sample_id, mask_filename)) if mask_filename else None
                if mask_filename and (not mask_path or not os.path.exists(mask_path)):
                    print(f"Skipping region with missing mask: {mask_path}")
                    continue
                label = region.get("semantic_label", "region")
                question = (
                    f"Assess the exposure quality of the {label} region and segment that region."
                )
                text = self._compose_dataset_example_text(region, global_saliency, global_strategy)
                rule_context = self._build_rule_context({
                    "grounding_records": [self._region_to_grounding_record(region)]
                }, question)
                segmentation_paths = [mask_path] if mask_path and os.path.exists(mask_path) else None
                samples.append(SamplePaths(image_path, mask_path, question, rule_context, text, segmentation_paths=segmentation_paths))

        if not samples:
            raise ValueError(f"No region samples were found in: {analysis_root}")
        return samples

    def _compose_dataset_example_text(
        self,
        region: Dict[str, object],
        global_saliency: Dict[str, object],
        global_strategy: Dict[str, object],
    ) -> str:
        label = region.get("semantic_label", "region")
        trend = region.get("exposure_trend", "Unknown exposure trend")
        score = region.get("exposure_score", "N/A")
        comment = region.get("exposure_comment", "")
        local_suggestion = region.get("local_adjustment_suggestion", "")
        primary_reason = global_saliency.get("primary_distortion_reason", "")
        global_suggestion = global_strategy.get("global_exposure_suggestion", "")
        balance_strategy = global_strategy.get("regional_balance_strategy", "")
        return (
            f"Region: {label}. Exposure trend: {trend}. Exposure score: {score}. "
            f"Regional analysis: {comment} Local adjustment: {local_suggestion} "
            f"Global saliency: {primary_reason} Global exposure strategy: {global_suggestion} "
            f"Regional balance strategy: {balance_strategy}"
        )

    def _region_to_grounding_record(self, region: Dict[str, object]) -> Dict[str, object]:
        trend = str(region.get("exposure_trend", ""))
        score = region.get("exposure_score")
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None

        record = {
            "label": region.get("semantic_label", "region"),
            "average_exposure_score": score_value,
            "exposure_trend": trend,
        }
        if "under" in trend.lower():
            record["exposure_std"] = 0.3
        elif "over" in trend.lower():
            record["exposure_std"] = 0.3
        return record

    def build_tokenizer(self, min_freq: int = 1, max_vocab: int = 12000) -> SimpleTokenizer:
        self.tokenizer.build(
            [self.compose_question(sample.question, sample.rule_context) for sample in self.samples]
            + [sample.text for sample in self.samples],
            min_freq=min_freq,
            max_vocab=max_vocab,
        )
        return self.tokenizer

    def _build_rule_context(self, item: Dict[str, object], question: str) -> str:
        if item.get("rule_context"):
            return str(item["rule_context"])

        if item.get("rules"):
            rules = item["rules"]
            if not isinstance(rules, list):
                raise ValueError("'rules' must be a list when provided")
            return RuleRAGRetriever.format_rules_for_prompt(rules)

        grounding_records = item.get("grounding_records") or []
        if not isinstance(grounding_records, list):
            raise ValueError("'grounding_records' must be a list when provided")

        retrieved_rules = self.rule_rag.retrieve(
            user_query=question,
            grounding_records=grounding_records,
            top_k=self.rag_top_k,
        )
        return RuleRAGRetriever.format_rules_for_prompt(retrieved_rules)

    @staticmethod
    def compose_question(question: str, rule_context: str) -> str:
        if not rule_context.strip():
            return question
        return f"{question}\n\nRetrieved exposure rules:\n{rule_context}"

    def _resolve(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(self.image_root, path)

    def _resolve_image(self, path: Optional[str]) -> Optional[str]:
        resolved = self._resolve(path)
        if not resolved:
            return None
        if os.path.exists(resolved):
            return resolved
        base, ext = os.path.splitext(resolved)
        if ext:
            return resolved
        for candidate_ext in self.image_extensions:
            candidate = resolved + candidate_ext
            if os.path.exists(candidate):
                return candidate
        return resolved

    def _load_image_tensor(self, sample: SamplePaths, image_path: Optional[str]) -> torch.Tensor:
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        elif sample.segmentation_paths:
            image = self._compose_segmentations(sample.segmentation_paths).convert("RGB")
        else:
            raise FileNotFoundError(
                f"Input image not found: {image_path}. Provide source images or segmentation_paths for pseudo input."
            )
        return self.image_transform(image)

    def _load_mask_tensor(self, sample: SamplePaths, mask_path: Optional[str]) -> torch.Tensor:
        if sample.segmentation_paths:
            mask = self._build_union_mask(sample.segmentation_paths)
            return self.mask_transform(mask)

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            return (mask_tensor > 0.5).float()

        return torch.zeros(1, self.image_size, self.image_size)

    def _compose_segmentations(self, segmentation_paths: List[str]) -> Image.Image:
        canvas: Optional[Image.Image] = None
        for path in segmentation_paths:
            segment = Image.open(path).convert("RGBA")
            if canvas is None:
                canvas = Image.new("RGBA", segment.size, (0, 0, 0, 255))
            canvas.alpha_composite(segment)
        if canvas is None:
            raise ValueError("No segmentation images were provided")
        return canvas

    def _build_union_mask(self, segmentation_paths: List[str]) -> Image.Image:
        union_mask: Optional[Image.Image] = None
        for path in segmentation_paths:
            segment = Image.open(path).convert("RGBA")
            if segment.getextrema()[3][1] > 0:
                mask = segment.getchannel("A")
            else:
                mask = segment.convert("L").point(lambda value: 255 if value > 5 else 0)
            mask = mask.point(lambda value: 255 if value > 0 else 0)
            if union_mask is None:
                union_mask = Image.new("L", mask.size, 0)
            union_mask = Image.composite(Image.new("L", mask.size, 255), union_mask, mask)
        if union_mask is None:
            return Image.new("L", (self.image_size, self.image_size), 0)
        return union_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_path = self._resolve_image(sample.image_path)
        if image_path is None:
            raise ValueError("Image path is missing")

        image_tensor = self._load_image_tensor(sample, image_path)

        mask_path = self._resolve(sample.mask_path)
        mask_tensor = self._load_mask_tensor(sample, mask_path)

        token_ids = self.tokenizer.encode(sample.text, self.max_text_len)
        question_with_rules = self.compose_question(sample.question, sample.rule_context)
        question_ids = self.tokenizer.encode(question_with_rules, self.max_question_len)
        decoder_input = token_ids[:-1]
        decoder_target = token_ids[1:]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "question_ids": question_ids,
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
        }


class DatasetExampleDataset(ReasoningGroundingDataset):
    """
    Supports the provided Dataset Example directory:

    Dataset Example/
      analysis_only/<sample_id>.json
      segmentations/<sample_id>/<region_mask_pngs>

    Source images are optional for format checking. If a source image is not
    found, the dataset composes the segmented PNGs into a pseudo input image.
    """

    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

    def __init__(
        self,
        dataset_root: str,
        source_image_root: str = "",
        tokenizer: Optional[SimpleTokenizer] = None,
        image_size: int = 256,
        max_text_len: int = 128,
        max_question_len: int = 64,
        rules_path: str = "",
        rag_top_k: int = 5,
    ):
        self.dataset_root = dataset_root
        self.source_image_root = source_image_root
        super().__init__(
            jsonl_path="",
            image_root=source_image_root or dataset_root,
            tokenizer=tokenizer,
            image_size=image_size,
            max_text_len=max_text_len,
            max_question_len=max_question_len,
            rules_path=rules_path,
            rag_top_k=rag_top_k,
        )

    def _load_samples(self, jsonl_path: str) -> List[SamplePaths]:
        analysis_dir = os.path.join(self.dataset_root, "analysis_only")
        segmentation_root = os.path.join(self.dataset_root, "segmentations")
        if not os.path.isdir(analysis_dir):
            raise FileNotFoundError(f"Missing analysis_only directory: {analysis_dir}")
        if not os.path.isdir(segmentation_root):
            raise FileNotFoundError(f"Missing segmentations directory: {segmentation_root}")

        samples: List[SamplePaths] = []
        for filename in sorted(os.listdir(analysis_dir)):
            if not filename.lower().endswith(".json"):
                continue
            sample_id = os.path.splitext(filename)[0]
            json_path = os.path.join(analysis_dir, filename)
            with open(json_path, "r", encoding="utf-8") as f:
                item = json.load(f)

            segmentation_dir = os.path.join(segmentation_root, sample_id)
            segmentation_paths = self._collect_segmentation_paths(segmentation_dir, item)
            source_image_path = self._find_source_image(sample_id)
            question = self._build_dataset_question(item)
            grounding_records = self._build_grounding_records(item)
            text = self._analysis_json_to_text(item)
            rule_context = RuleRAGRetriever.format_rules_for_prompt(
                self.rule_rag.retrieve(question, grounding_records, top_k=self.rag_top_k)
            )

            samples.append(
                SamplePaths(
                    image_path=source_image_path or os.path.join(self.source_image_root or self.dataset_root, f"{sample_id}.jpg"),
                    mask_path=None,
                    question=question,
                    rule_context=rule_context,
                    text=text,
                    segmentation_paths=segmentation_paths,
                )
            )
        if not samples:
            raise ValueError(f"No JSON analysis files found in: {analysis_dir}")
        return samples

    def _collect_segmentation_paths(self, segmentation_dir: str, item: Dict[str, object]) -> List[str]:
        if not os.path.isdir(segmentation_dir):
            raise FileNotFoundError(f"Missing segmentation directory: {segmentation_dir}")

        region_files = []
        for region in item.get("semantic_regions", []):
            if isinstance(region, dict) and region.get("mask_filename"):
                path = os.path.join(segmentation_dir, str(region["mask_filename"]))
                if os.path.exists(path):
                    region_files.append(path)

        if region_files:
            return region_files

        return [
            os.path.join(segmentation_dir, name)
            for name in sorted(os.listdir(segmentation_dir))
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

    def _find_source_image(self, sample_id: str) -> Optional[str]:
        search_roots = []
        if self.source_image_root:
            search_roots.append(self.source_image_root)
        search_roots.extend([
            os.path.join(self.dataset_root, "images"),
            os.path.join(self.dataset_root, "source_images"),
            self.dataset_root,
        ])

        for root in search_roots:
            if not root or not os.path.isdir(root):
                continue
            for ext in self.IMAGE_EXTENSIONS:
                path = os.path.join(root, sample_id + ext)
                if os.path.exists(path):
                    return path
        return None

    def _build_dataset_question(self, item: Dict[str, object]) -> str:
        labels = []
        for region in item.get("semantic_regions", []):
            if isinstance(region, dict) and region.get("semantic_label"):
                labels.append(str(region["semantic_label"]))
        if labels:
            return "Assess exposure quality and segment these regions: " + ", ".join(labels) + "."
        return self.DEFAULT_QUESTION

    def _build_grounding_records(self, item: Dict[str, object]) -> List[Dict[str, object]]:
        records = []
        for region in item.get("semantic_regions", []):
            if not isinstance(region, dict):
                continue
            records.append({
                "label": region.get("semantic_label", "region"),
                "average_exposure_score": region.get("exposure_score"),
                "exposure_trend": region.get("exposure_trend"),
                "mask_path": region.get("mask_filename"),
            })
        return records

    def _analysis_json_to_text(self, item: Dict[str, object]) -> str:
        parts: List[str] = []
        for region in item.get("semantic_regions", []):
            if not isinstance(region, dict):
                continue
            label = region.get("semantic_label", f"region {region.get('region_id', '')}")
            trend = region.get("exposure_trend", "unknown exposure")
            score = region.get("exposure_score", "unknown")
            comment = region.get("exposure_comment", "")
            suggestion = region.get("local_adjustment_suggestion", "")
            parts.append(
                f"Region {region.get('region_id', '?')} ({label}) has trend {trend} "
                f"with exposure score {score}. {comment} Suggestion: {suggestion}"
            )

        saliency = item.get("global_saliency_analysis", {})
        if isinstance(saliency, dict):
            reason = saliency.get("primary_distortion_reason")
            if reason:
                parts.append(f"Primary distortion: {reason}")

        strategy = item.get("global_exposure_strategy", {})
        if isinstance(strategy, dict):
            global_suggestion = strategy.get("global_exposure_suggestion")
            regional_strategy = strategy.get("regional_balance_strategy")
            if global_suggestion:
                parts.append(f"Global exposure suggestion: {global_suggestion}")
            if regional_strategy:
                parts.append(f"Regional balance strategy: {regional_strategy}")

        return " ".join(str(part).strip() for part in parts if str(part).strip())


class VisualEncoder(nn.Module):
    """Converts an input image into dense visual tokens plus a feature map for segmentation."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.token_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(image)
        visual_tokens = feature_map.flatten(2).transpose(1, 2)
        return feature_map, self.token_norm(visual_tokens)


class FrozenPIEANetHeatmap(nn.Module):
    """Frozen P-IEANet wrapper that predicts exposure heatmaps for the quality branch."""

    def __init__(self, model_path: str = ""):
        super().__init__()
        project_dir = os.path.dirname(os.path.abspath(__file__))
        pieanet_dir = os.path.join(project_dir, "P-IEANet")
        if not model_path:
            model_path = os.path.join(pieanet_dir, "wavelet_epoch_2_val_loss0.03022034629540784.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"P-IEANet weight not found: {model_path}")

        if pieanet_dir not in sys.path:
            sys.path.append(pieanet_dir)
        from wavelet_network import Wavelet_Net

        self.model = Wavelet_Net()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model_path = model_path

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        original_size = image.shape[-2:]
        pieanet_input = F.interpolate(image.float(), size=(256, 256), mode="bilinear", align_corners=False)
        doubled = torch.cat([pieanet_input, pieanet_input], dim=0)
        exposure = self.model(doubled)[: image.shape[0]]
        exposure = F.interpolate(exposure, size=original_size, mode="bilinear", align_corners=False)
        return exposure * -1.0


class QualityEncoder(nn.Module):
    """
    Encodes the frozen P-IEANet exposure heatmap into compact quality tokens.
    """

    def __init__(self, hidden_dim: int = 256, token_grid: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))
        self.token_norm = nn.LayerNorm(hidden_dim)

    def forward(self, exposure_heatmap: torch.Tensor) -> torch.Tensor:
        if exposure_heatmap.ndim == 3:
            exposure_heatmap = exposure_heatmap.unsqueeze(1)
        quality_map = self.encoder(exposure_heatmap.float())
        quality_tokens = self.pool(quality_map).flatten(2).transpose(1, 2)
        return self.token_norm(quality_tokens)


class QuestionEncoder(nn.Module):
    """Encodes question tokens before multimodal fusion."""

    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.position = nn.Embedding(256, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(question_ids.shape[1], device=question_ids.device).unsqueeze(0)
        pad_mask = question_ids.eq(self.pad_id)
        tokens = self.embedding(question_ids) + self.position(positions)
        tokens = self.encoder(tokens, src_key_padding_mask=pad_mask)
        return self.norm(tokens), pad_mask


class LLMBackbone(nn.Module):
    """Compact Transformer backbone for visual, quality, and question token fusion."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        self.position = nn.Embedding(max_tokens, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        quality_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        question_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([visual_tokens, quality_tokens, question_tokens], dim=1)
        batch_size = tokens.shape[0]
        visual_len = visual_tokens.shape[1]
        quality_len = quality_tokens.shape[1]
        question_len = question_tokens.shape[1]

        type_ids = torch.cat([
            torch.zeros(visual_len, dtype=torch.long, device=tokens.device),
            torch.ones(quality_len, dtype=torch.long, device=tokens.device),
            torch.full((question_len,), 2, dtype=torch.long, device=tokens.device),
        ]).unsqueeze(0).expand(batch_size, -1)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        tokens = tokens + self.position(positions) + self.type_embedding(type_ids)

        prefix_pad = torch.zeros(batch_size, visual_len + quality_len, dtype=torch.bool, device=tokens.device)
        memory_pad_mask = torch.cat([prefix_pad, question_pad_mask], dim=1)
        memory = self.encoder(tokens, src_key_padding_mask=memory_pad_mask)
        return self.norm(memory), memory_pad_mask


class TextGenerationHead(nn.Module):
    """Autoregressive text head that decodes the final report from fused tokens."""

    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.position = nn.Embedding(256, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_input: torch.Tensor, memory: torch.Tensor, memory_pad_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(decoder_input.shape[1], device=decoder_input.device).unsqueeze(0)
        target = self.embedding(decoder_input) + self.position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1], device=decoder_input.device)
        target_pad_mask = decoder_input.eq(self.pad_id)
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.output(decoded)


class SegmentationHead(nn.Module):
    """Mask head conditioned by the multimodal backbone state."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.condition = nn.Linear(hidden_dim, hidden_dim)
        mid_dim = max(hidden_dim // 2, 32)
        low_dim = max(hidden_dim // 4, 16)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, low_dim, 3, padding=1),
            nn.BatchNorm2d(low_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_dim, 1, 1),
        )

    def forward(
        self,
        visual_map: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        valid = (~memory_pad_mask).float().unsqueeze(-1)
        pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        condition = self.condition(pooled).unsqueeze(-1).unsqueeze(-1)
        conditioned_map = visual_map + condition
        seg_logits = self.decoder(conditioned_map)
        return F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)


class LlamaReasoningGroundingModel(nn.Module):
    """
    Uses a pretrained Llama-style causal LM as the reasoning/text backbone.

    Visual and quality tokens are projected to the Llama embedding space and
    prepended as soft prefix tokens. The Llama hidden states then condition both
    report generation and the segmentation mask head.
    """

    def __init__(
        self,
        llama_model_name: str,
        hidden_dim: int = 256,
        question_vocab_size: int = 32000,
        question_pad_id: int = 0,
        num_layers: int = 4,
        num_heads: int = 4,
        freeze_llama: bool = False,
        torch_dtype: str = "auto",
        local_files_only: bool = False,
        pieanet_model_path: str = "",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        dtype = None
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32

        model_kwargs = {
            "local_files_only": local_files_only,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.llama_model_name = llama_model_name
        self.hidden_dim = hidden_dim
        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name, **model_kwargs)
        self.llama.config.use_cache = False
        self.use_lora = use_lora
        if use_lora:
            from peft import LoraConfig, get_peft_model

            target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama = get_peft_model(self.llama, lora_config)
        if freeze_llama:
            for param in self.llama.parameters():
                param.requires_grad = False

        llama_dim = int(self.llama.config.hidden_size)
        self.pieanet_heatmap = FrozenPIEANetHeatmap(pieanet_model_path)
        self.visual_encoder = VisualEncoder(hidden_dim=hidden_dim)
        self.quality_encoder = QualityEncoder(hidden_dim=hidden_dim)
        self.question_encoder = QuestionEncoder(
            vocab_size=question_vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers // 2),
            num_heads=num_heads,
            pad_id=question_pad_id,
        )
        self.prefix_to_llama = nn.Linear(hidden_dim, llama_dim)
        self.question_to_llama = nn.Linear(hidden_dim, llama_dim)
        self.llama_to_hidden = nn.Linear(llama_dim, hidden_dim)
        self.segmentation_head = SegmentationHead(hidden_dim=hidden_dim)

    def _embed_inputs(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        visual_map, visual_tokens = self.visual_encoder(image)
        exposure_heatmap = self.pieanet_heatmap(image)
        quality_tokens = self.quality_encoder(exposure_heatmap)
        prefix_tokens = torch.cat([visual_tokens, quality_tokens], dim=1)
        prefix_embeds = self.prefix_to_llama(prefix_tokens).to(dtype=self.llama.get_input_embeddings().weight.dtype)

        question_tokens, question_pad_mask = self.question_encoder(question_ids)
        question_embeds = self.question_to_llama(question_tokens).to(dtype=self.llama.get_input_embeddings().weight.dtype)
        embeds = [prefix_embeds, question_embeds]
        masks = [
            torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=image.device),
            (~question_pad_mask).long(),
        ]
        if decoder_input is not None:
            decoder_mask = decoder_input.ne(self.llama.config.pad_token_id or 0)
            decoder_embeds = self.llama.get_input_embeddings()(decoder_input)
            embeds.append(decoder_embeds)
            masks.append(decoder_mask.long())

        inputs_embeds = torch.cat(embeds, dim=1)
        attention_mask = torch.cat(masks, dim=1)
        return visual_map, inputs_embeds, attention_mask, prefix_embeds.shape[1], question_embeds.shape[1]

    def encode_context(self, image: torch.Tensor, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_map, inputs_embeds, attention_mask, _, _ = self._embed_inputs(image, question_ids)
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        memory = self.llama_to_hidden(outputs.hidden_states[-1].float())
        memory_pad_mask = attention_mask.eq(0)
        return visual_map, memory, memory_pad_mask

    def forward(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_map, inputs_embeds, attention_mask, prefix_len, question_len = self._embed_inputs(
            image,
            question_ids,
            decoder_input,
        )
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        memory = self.llama_to_hidden(hidden.float())
        memory_pad_mask = attention_mask.eq(0)
        seg_logits = self.segmentation_head(visual_map, memory, memory_pad_mask, output_size=image.shape[-2:])
        decoder_start = prefix_len + question_len
        text_logits = outputs.logits[:, decoder_start:, :]
        return seg_logits, text_logits

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        tokenizer,
        max_len: int = 96,
    ) -> Tuple[torch.Tensor, List[int]]:
        self.eval()
        visual_map, memory, memory_pad_mask = self.encode_context(image, question_ids)
        seg_logits = self.segmentation_head(visual_map, memory, memory_pad_mask, output_size=image.shape[-2:])
        current = torch.tensor([[tokenizer.bos_id]], device=image.device)
        generated: List[int] = []
        for _ in range(max_len):
            _, logits = self.forward(image, question_ids, current)
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if next_id == tokenizer.eos_id:
                break
            generated.append(next_id)
            current = torch.cat([current, torch.tensor([[next_id]], device=image.device)], dim=1)
        return seg_logits, generated


class ReasoningGroundingModel(nn.Module):
    """
    Similar-in-spirit architecture to the paper:
    Quality Encoder + Visual Encoder + Question Encoder -> LLM Backbone -> text and mask heads.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        pad_id: int = 0,
        max_tokens: int = 1024,
        pieanet_model_path: str = "",
    ):
        super().__init__()
        self.pieanet_heatmap = FrozenPIEANetHeatmap(pieanet_model_path)
        self.visual_encoder = VisualEncoder(hidden_dim=hidden_dim)
        self.quality_encoder = QualityEncoder(hidden_dim=hidden_dim)
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers // 2),
            num_heads=num_heads,
            pad_id=pad_id,
        )
        self.llm_backbone = LLMBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_tokens=max_tokens,
        )
        self.text_head = TextGenerationHead(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers // 2),
            num_heads=num_heads,
            pad_id=pad_id,
        )
        self.segmentation_head = SegmentationHead(hidden_dim=hidden_dim)

    def encode_context(self, image: torch.Tensor, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_map, visual_tokens = self.visual_encoder(image)
        exposure_heatmap = self.pieanet_heatmap(image)
        quality_tokens = self.quality_encoder(exposure_heatmap)
        question_tokens, question_pad_mask = self.question_encoder(question_ids)
        memory, memory_pad_mask = self.llm_backbone(
            visual_tokens=visual_tokens,
            quality_tokens=quality_tokens,
            question_tokens=question_tokens,
            question_pad_mask=question_pad_mask,
        )
        return visual_map, memory, memory_pad_mask

    def forward(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_map, memory, memory_pad_mask = self.encode_context(image, question_ids)
        seg_logits = self.segmentation_head(visual_map, memory, memory_pad_mask, output_size=image.shape[-2:])
        text_logits = self.text_head(decoder_input, memory, memory_pad_mask)
        return seg_logits, text_logits

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        tokenizer: SimpleTokenizer,
        max_len: int = 96,
    ) -> Tuple[torch.Tensor, List[int]]:
        self.eval()
        visual_map, memory, memory_pad_mask = self.encode_context(image, question_ids)
        seg_logits = self.segmentation_head(visual_map, memory, memory_pad_mask, output_size=image.shape[-2:])
        current = torch.tensor([[tokenizer.bos_id]], device=image.device)
        generated: List[int] = []

        for _ in range(max_len):
            logits = self.text_head(current, memory, memory_pad_mask)
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if next_id == tokenizer.eos_id:
                break
            generated.append(next_id)
            current = torch.cat([current, torch.tensor([[next_id]], device=image.device)], dim=1)

        return seg_logits, generated


def parse_thresholds(value: str, fallback: float) -> List[float]:
    if not value:
        return [fallback]
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        return [fallback]
    for threshold in thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("--eval-thresholds values must be between 0 and 1")
    return sorted(set(thresholds))


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob_flat = prob.flatten(1)
    target_flat = target.flatten(1)
    inter = (prob_flat * target_flat).sum(dim=1)
    denom = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def segmentation_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pos_weight = None
    if args.auto_pos_weight:
        positive = target.sum().clamp_min(1.0)
        negative = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (negative / positive).clamp(max=args.max_pos_weight).to(logits.device)
    elif args.pos_weight > 0:
        pos_weight = torch.tensor(args.pos_weight, device=logits.device, dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, target)
    focal = focal_loss_from_logits(logits, target, alpha=args.focal_alpha, gamma=args.focal_gamma)

    if args.seg_loss_type == "bce":
        loss = bce
    elif args.seg_loss_type == "bce_dice":
        loss = args.bce_loss_weight * bce + args.dice_loss_weight * dice
    elif args.seg_loss_type == "bce_focal_dice":
        loss = args.bce_loss_weight * bce + args.focal_loss_weight * focal + args.dice_loss_weight * dice
    else:
        raise ValueError(f"Unsupported seg loss type: {args.seg_loss_type}")

    stats = {
        "loss_bce": float(bce.detach().item()),
        "loss_dice": float(dice.detach().item()),
        "loss_focal": float(focal.detach().item()),
        "pos_weight": float(pos_weight.detach().item()) if pos_weight is not None else 0.0,
    }
    return loss, stats


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.train_jsonl and not args.dataset_root:
        raise ValueError("Provide --train-jsonl for JSONL data or --dataset-root for Dataset Example data")

    tokenizer = None
    if args.backbone_type == "llama":
        if not args.llama_model_name:
            raise ValueError("--llama-model-name is required when --backbone-type llama is used")
        tokenizer_name = args.llama_tokenizer_name or args.llama_model_name
        tokenizer = HFTokenizerWrapper.from_pretrained(
            tokenizer_name,
            local_files_only=args.local_files_only,
        )

    dataset = ReasoningGroundingDataset(
        args.train_jsonl,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
        max_question_len=args.max_question_len,
        rules_path=args.rules_path,
        rag_top_k=args.rag_top_k,
        dataset_format=args.dataset_format,
        dataset_root=args.dataset_root,
        analysis_dir=args.analysis_dir,
        segmentation_dir=args.segmentation_dir,
        image_extensions=args.image_extensions,
    )
    tokenizer = dataset.build_tokenizer(min_freq=args.min_freq, max_vocab=args.max_vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = None
    if args.eval_dataset_root or args.eval_jsonl:
        eval_dataset = ReasoningGroundingDataset(
            args.eval_jsonl,
            image_root=args.image_root,
            tokenizer=tokenizer,
            image_size=args.image_size,
            max_text_len=args.max_text_len,
            max_question_len=args.max_question_len,
            rules_path=args.rules_path,
            rag_top_k=args.rag_top_k,
            dataset_format=args.eval_dataset_format or args.dataset_format,
            dataset_root=args.eval_dataset_root,
            analysis_dir=args.analysis_dir,
            segmentation_dir=args.segmentation_dir,
            image_extensions=args.image_extensions,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    if args.backbone_type == "llama":
        model = LlamaReasoningGroundingModel(
            llama_model_name=args.llama_model_name,
            hidden_dim=args.hidden_dim,
            question_vocab_size=len(tokenizer),
            question_pad_id=tokenizer.pad_id,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            freeze_llama=args.freeze_llama,
            torch_dtype=args.llama_torch_dtype,
            local_files_only=args.local_files_only,
            pieanet_model_path=args.pieanet_model_path,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        )
        model.llama.config.pad_token_id = tokenizer.pad_id
    else:
        model = ReasoningGroundingModel(
            vocab_size=len(tokenizer),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            pad_id=tokenizer.pad_id,
            pieanet_model_path=args.pieanet_model_path,
        )
    start_epoch = 0
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(resume["model_state"], strict=False)
        start_epoch = int(resume.get("epoch", 0))
        print(
            f"resumed_from={args.resume_checkpoint} start_epoch={start_epoch} "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
    model = model.to(device)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"trainable_params={trainable_params} total_params={total_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_jsonl = os.path.join(args.output_dir, "epoch_metrics.jsonl")
    previous_loss: Optional[float] = None
    stable_epochs = 0
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_text_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        total_focal = 0.0
        total_pos_weight = 0.0
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            question_ids = batch["question_ids"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)

            seg_logits, text_logits = model(image, question_ids, decoder_input)
            loss_seg, seg_stats = segmentation_loss_from_logits(seg_logits, mask, args)
            loss_text = ce_loss(text_logits.reshape(-1, text_logits.shape[-1]), decoder_target.reshape(-1))
            loss = args.seg_loss_weight * loss_seg + args.text_loss_weight * loss_text

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_seg_loss += float(loss_seg.item())
            total_text_loss += float(loss_text.item())
            total_bce += seg_stats["loss_bce"]
            total_dice += seg_stats["loss_dice"]
            total_focal += seg_stats["loss_focal"]
            total_pos_weight += seg_stats["pos_weight"]

        avg_loss = total_loss / max(len(loader), 1)
        avg_seg_loss = total_seg_loss / max(len(loader), 1)
        avg_text_loss = total_text_loss / max(len(loader), 1)
        avg_bce = total_bce / max(len(loader), 1)
        avg_dice = total_dice / max(len(loader), 1)
        avg_focal = total_focal / max(len(loader), 1)
        avg_pos_weight = total_pos_weight / max(len(loader), 1)
        print(
            f"epoch={epoch} loss={avg_loss:.4f} "
            f"seg_loss={avg_seg_loss:.4f} text_loss={avg_text_loss:.4f} "
            f"bce={avg_bce:.4f} dice={avg_dice:.4f} focal={avg_focal:.4f} "
            f"pos_weight={avg_pos_weight:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "image_size": args.image_size,
            "max_text_len": args.max_text_len,
            "max_question_len": args.max_question_len,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "rules_path": args.rules_path,
            "rag_top_k": args.rag_top_k,
            "backbone_type": args.backbone_type,
            "llama_model_name": args.llama_model_name,
            "llama_tokenizer_name": args.llama_tokenizer_name,
            "freeze_llama": args.freeze_llama,
            "llama_torch_dtype": args.llama_torch_dtype,
            "local_files_only": args.local_files_only,
            "pieanet_model_path": args.pieanet_model_path,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "seg_loss_type": args.seg_loss_type,
            "bce_loss_weight": args.bce_loss_weight,
            "dice_loss_weight": args.dice_loss_weight,
            "focal_loss_weight": args.focal_loss_weight,
            "pos_weight": args.pos_weight,
            "auto_pos_weight": args.auto_pos_weight,
            "max_pos_weight": args.max_pos_weight,
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_seg_loss": avg_seg_loss,
            "train_text_loss": avg_text_loss,
            "train_bce_loss": avg_bce,
            "train_dice_loss": avg_dice,
            "train_focal_loss": avg_focal,
            "train_pos_weight": avg_pos_weight,
            "architecture": (
                "frozen_pieanet_heatmap_quality_visual_question_encoder_to_llama_backbone"
                if args.backbone_type == "llama"
                else "frozen_pieanet_heatmap_quality_visual_question_encoder_to_llm_backbone"
            ),
        }
        if args.backbone_type == "llama":
            checkpoint["tokenizer"] = {
                "type": "huggingface",
                "model_name_or_path": args.llama_tokenizer_name or args.llama_model_name,
            }
        latest_path = os.path.join(args.output_dir, "reasoning_grounding_model.pt")
        epoch_path = os.path.join(args.output_dir, f"reasoning_grounding_model_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, epoch_path)
        if os.path.exists(latest_path):
            os.remove(latest_path)
        try:
            os.link(epoch_path, latest_path)
        except OSError:
            torch.save(checkpoint, latest_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_seg_loss": avg_seg_loss,
            "train_text_loss": avg_text_loss,
            "train_bce_loss": avg_bce,
            "train_dice_loss": avg_dice,
            "train_focal_loss": avg_focal,
            "train_pos_weight": avg_pos_weight,
            "checkpoint": epoch_path,
        }
        if eval_loader is not None:
            eval_metrics = _evaluate_model(
                model=model,
                tokenizer=tokenizer,
                loader=eval_loader,
                device=device,
                threshold=args.eval_threshold,
                thresholds=parse_thresholds(args.eval_thresholds, args.eval_threshold),
                max_batches=args.eval_max_batches,
            )
            metrics_path = os.path.join(args.output_dir, f"eval_metrics_epoch_{epoch:03d}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
            epoch_record.update(eval_metrics)
            epoch_record["eval_metrics_path"] = metrics_path
            print(
                f"epoch={epoch} BLEU@4={eval_metrics['BLEU@4']:.4f} "
                f"mIoU={eval_metrics['mIoU']:.4f} mAcc={eval_metrics['mAcc']:.4f}"
            )

        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        if previous_loss is not None and args.early_stop_min_delta > 0:
            if abs(previous_loss - avg_loss) < args.early_stop_min_delta:
                stable_epochs += 1
            else:
                stable_epochs = 0
            if stable_epochs >= args.early_stop_patience:
                print(
                    f"early_stop=loss_converged epoch={epoch} "
                    f"previous_loss={previous_loss:.6f} current_loss={avg_loss:.6f} "
                    f"min_delta={args.early_stop_min_delta} patience={args.early_stop_patience}"
                )
                break
        previous_loss = avg_loss


def load_checkpoint(path: str, device: torch.device) -> Tuple[ReasoningGroundingModel, SimpleTokenizer, int, int, int]:
    checkpoint = torch.load(path, map_location=device)
    backbone_type = checkpoint.get("backbone_type", "compact")
    if backbone_type == "llama":
        tokenizer_info = checkpoint.get("tokenizer", {})
        model_name = checkpoint.get("llama_model_name") or tokenizer_info.get("model_name_or_path")
        tokenizer_name = (
            checkpoint.get("llama_tokenizer_name")
            or tokenizer_info.get("model_name_or_path")
            or model_name
        )
        if not model_name:
            raise ValueError("Llama checkpoint is missing llama_model_name")
        model_name = resolve_packaged_model_path(model_name, path)
        tokenizer_name = resolve_packaged_model_path(tokenizer_name, path)
        tokenizer = HFTokenizerWrapper.from_pretrained(
            tokenizer_name,
            local_files_only=checkpoint.get("local_files_only", False),
        )
        model = LlamaReasoningGroundingModel(
            llama_model_name=model_name,
            hidden_dim=checkpoint.get("hidden_dim", 256),
            question_vocab_size=len(tokenizer),
            question_pad_id=tokenizer.pad_id,
            num_layers=checkpoint.get("num_layers", 4),
            num_heads=checkpoint.get("num_heads", 4),
            freeze_llama=checkpoint.get("freeze_llama", False),
            torch_dtype=checkpoint.get("llama_torch_dtype", "auto"),
            local_files_only=checkpoint.get("local_files_only", False),
            pieanet_model_path=checkpoint.get("pieanet_model_path", ""),
            use_lora=checkpoint.get("use_lora", False),
            lora_r=checkpoint.get("lora_r", 8),
            lora_alpha=checkpoint.get("lora_alpha", 16),
            lora_dropout=checkpoint.get("lora_dropout", 0.05),
            lora_target_modules=checkpoint.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj"),
        )
        model.llama.config.pad_token_id = tokenizer.pad_id
    else:
        tokenizer = SimpleTokenizer(checkpoint["tokenizer"])
        model = ReasoningGroundingModel(
            vocab_size=len(tokenizer),
            hidden_dim=checkpoint.get("hidden_dim", 256),
            num_layers=checkpoint.get("num_layers", 4),
            num_heads=checkpoint.get("num_heads", 4),
            pad_id=tokenizer.pad_id,
            pieanet_model_path=checkpoint.get("pieanet_model_path", ""),
        )
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print(
            f"checkpoint_loaded_with_partial_match missing_keys={len(missing)} "
            f"unexpected_keys={len(unexpected)}"
        )
    model.to(device)
    model.eval()
    return (
        model,
        tokenizer,
        checkpoint.get("image_size", 256),
        checkpoint.get("max_text_len", 128),
        checkpoint.get("max_question_len", 64),
    )


def resolve_packaged_model_path(model_name_or_path: str, checkpoint_path: str) -> str:
    if not model_name_or_path:
        return model_name_or_path
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    env_override = os.environ.get("LLAMA_MODEL_NAME") or os.environ.get("LLAMA_MODEL_PATH")
    if env_override and os.path.exists(env_override):
        return env_override

    checkpoint_dir = os.path.abspath(os.path.dirname(checkpoint_path))
    candidates = []
    current = checkpoint_dir
    for _ in range(5):
        candidates.extend([
            os.path.join(current, "llama8b_base"),
            os.path.join(current, "NousResearch-Meta-Llama-3-8B"),
            os.path.join(current, "weights", "llama8b_base"),
            os.path.join(current, "weights", "NousResearch-Meta-Llama-3-8B"),
        ])
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return model_name_or_path


def save_segmented_image(image_path: str, mask_logits: torch.Tensor, output_path: str, threshold: float = 0.5) -> None:
    image = Image.open(image_path).convert("RGBA")
    original_size = image.size
    mask = torch.sigmoid(mask_logits[0, 0]).detach().cpu()
    mask_image = transforms.ToPILImage()(mask)
    mask_image = mask_image.resize(original_size, Image.Resampling.BILINEAR)
    alpha = mask_image.point(lambda value: 255 if value / 255.0 >= threshold else 0)
    image.putalpha(alpha)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, image_size, max_text_len, max_question_len = load_checkpoint(args.checkpoint, device)
    image = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    question = args.question or ReasoningGroundingDataset.DEFAULT_QUESTION
    rule_context = build_inference_rule_context(
        question=question,
        rules_path=args.rules_path,
        rule_context=args.rule_context,
        grounding_records_json=args.grounding_records_json,
        rag_top_k=args.rag_top_k,
    )
    question_with_rules = ReasoningGroundingDataset.compose_question(question, rule_context)
    question_ids = tokenizer.encode(question_with_rules, max_question_len).unsqueeze(0).to(device)

    seg_logits, ids = model.generate(image_tensor, question_ids, tokenizer, max_len=args.max_text_len or max_text_len)
    text = tokenizer.decode(ids)
    save_segmented_image(args.image, seg_logits, args.output_image, threshold=args.threshold)

    os.makedirs(os.path.dirname(args.output_text) or ".", exist_ok=True)
    with open(args.output_text, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"segmented image saved to: {args.output_image}")
    print(f"text saved to: {args.output_text}")


def build_inference_rule_context(
    question: str,
    rules_path: str = "",
    rule_context: str = "",
    grounding_records_json: str = "",
    rag_top_k: int = 5,
) -> str:
    if rule_context:
        return rule_context

    grounding_records = []
    if grounding_records_json:
        if os.path.exists(grounding_records_json):
            with open(grounding_records_json, "r", encoding="utf-8") as f:
                grounding_records = json.load(f)
        else:
            grounding_records = json.loads(grounding_records_json)
        if not isinstance(grounding_records, list):
            raise ValueError("grounding records must be a JSON list")

    retriever = RuleRAGRetriever(rules_path=rules_path or None)
    retrieved_rules = retriever.retrieve(
        user_query=question,
        grounding_records=grounding_records,
        top_k=rag_top_k,
    )
    return RuleRAGRetriever.format_rules_for_prompt(retrieved_rules)


def _metric_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu4(predictions: List[str], references: List[str]) -> Dict[str, float]:
    clipped = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = _metric_tokens(pred)
        ref_tokens = _metric_tokens(ref)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)
        for n in range(1, 5):
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts = _ngram_counts(ref_tokens, n)
            totals[n - 1] += max(sum(pred_counts.values()), 0)
            clipped[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())

    precisions = []
    for hit, total in zip(clipped, totals):
        precisions.append((hit + 1.0) / (total + 1.0))
    if pred_len == 0:
        bleu = 0.0
        bp = 0.0
    else:
        bp = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / max(pred_len, 1)))
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return {
        "BLEU@4": float(bleu),
        "brevity_penalty": float(bp),
        "precision_1": float(precisions[0]),
        "precision_2": float(precisions[1]),
        "precision_3": float(precisions[2]),
        "precision_4": float(precisions[3]),
        "prediction_tokens": float(pred_len),
        "reference_tokens": float(ref_len),
    }


def _evaluate_model(
    model: nn.Module,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    thresholds: Optional[Sequence[float]] = None,
    max_batches: int = 0,
) -> Dict[str, float]:
    predictions: List[str] = []
    references: List[str] = []
    thresholds = list(thresholds or [threshold])
    if threshold not in thresholds:
        thresholds.append(threshold)
    thresholds = sorted(set(thresholds))
    threshold_stats = {
        item: {"iou_sum": 0.0, "pixel_acc_sum": 0.0, "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
        for item in thresholds
    }
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            question_ids = batch["question_ids"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)

            seg_logits, text_logits = model(image, question_ids, decoder_input)
            pred_prob = torch.sigmoid(seg_logits)
            true_mask = mask >= 0.5

            batch_size = image.shape[0]
            true_flat = true_mask.view(batch_size, -1)
            n_samples += batch_size

            for item in thresholds:
                pred_flat = (pred_prob >= item).view(batch_size, -1)
                inter = (pred_flat & true_flat).sum(dim=1).float()
                union = (pred_flat | true_flat).sum(dim=1).float()
                sample_iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
                sample_acc = (pred_flat == true_flat).float().mean(dim=1)
                stats = threshold_stats[item]
                stats["iou_sum"] += float(sample_iou.sum().item())
                stats["pixel_acc_sum"] += float(sample_acc.sum().item())
                stats["tp"] += float((pred_flat & true_flat).sum().item())
                stats["fp"] += float((pred_flat & ~true_flat).sum().item())
                stats["tn"] += float((~pred_flat & ~true_flat).sum().item())
                stats["fn"] += float((~pred_flat & true_flat).sum().item())

            pred_ids = torch.argmax(text_logits, dim=-1).detach().cpu().tolist()
            target_ids = decoder_target.detach().cpu().tolist()
            for pred_seq, target_seq in zip(pred_ids, target_ids):
                predictions.append(tokenizer.decode(pred_seq))
                references.append(tokenizer.decode(target_seq))

            if max_batches and batch_idx >= max_batches:
                break

    bleu = corpus_bleu4(predictions, references)
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    best_threshold = threshold
    best_miou = -1.0
    for item in thresholds:
        stats = threshold_stats[item]
        tp = stats["tp"]
        fp = stats["fp"]
        tn = stats["tn"]
        fn = stats["fn"]
        fg_acc = tp / max(tp + fn, 1.0)
        bg_acc = tn / max(tn + fp, 1.0)
        item_metrics = {
            "threshold": item,
            "mIoU": stats["iou_sum"] / max(n_samples, 1),
            "pixel_accuracy": stats["pixel_acc_sum"] / max(n_samples, 1),
            "mAcc": 0.5 * (fg_acc + bg_acc),
            "foreground_accuracy": fg_acc,
            "background_accuracy": bg_acc,
            "mean_class_accuracy": 0.5 * (fg_acc + bg_acc),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        threshold_metrics[f"{item:.4f}"] = item_metrics
        if item_metrics["mIoU"] > best_miou:
            best_miou = item_metrics["mIoU"]
            best_threshold = item

    primary = threshold_metrics[f"{threshold:.4f}"]
    best = threshold_metrics[f"{best_threshold:.4f}"]
    metrics: Dict[str, float] = {
        "samples": n_samples,
        "threshold": threshold,
        "best_threshold": best_threshold,
        "BLEU@4": bleu["BLEU@4"],
        "mIoU": primary["mIoU"],
        "mAcc": primary["mAcc"],
        "pixel_accuracy": primary["pixel_accuracy"],
        "foreground_accuracy": primary["foreground_accuracy"],
        "background_accuracy": primary["background_accuracy"],
        "mean_class_accuracy": primary["mean_class_accuracy"],
        "best_mIoU": best["mIoU"],
        "best_mAcc": best["mAcc"],
        "best_pixel_accuracy": best["pixel_accuracy"],
        "tp": primary["tp"],
        "fp": primary["fp"],
        "tn": primary["tn"],
        "fn": primary["fn"],
        "threshold_metrics": threshold_metrics,
        **{f"text_{key}": value for key, value in bleu.items() if key != "BLEU@4"},
    }
    return metrics


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, image_size, max_text_len, max_question_len = load_checkpoint(args.checkpoint, device)
    dataset = ReasoningGroundingDataset(
        args.test_jsonl,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_size=args.image_size or image_size,
        max_text_len=args.max_text_len or max_text_len,
        max_question_len=args.max_question_len or max_question_len,
        rules_path=args.rules_path,
        rag_top_k=args.rag_top_k,
        dataset_format=args.dataset_format,
        dataset_root=args.dataset_root,
        analysis_dir=args.analysis_dir,
        segmentation_dir=args.segmentation_dir,
        image_extensions=args.image_extensions,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = _evaluate_model(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        threshold=args.threshold,
        thresholds=parse_thresholds(args.thresholds, args.threshold),
        max_batches=args.max_batches,
    )
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a reasoning-grounding model with quality, visual, and question encoders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-jsonl", default="")
    train_parser.add_argument("--resume-checkpoint", default="")
    train_parser.add_argument("--dataset-format", choices=["auto", "jsonl", "dataset_example"], default="auto")
    train_parser.add_argument("--dataset-root", default="")
    train_parser.add_argument("--eval-jsonl", default="")
    train_parser.add_argument("--eval-dataset-format", choices=["", "auto", "jsonl", "dataset_example"], default="")
    train_parser.add_argument("--eval-dataset-root", default="")
    train_parser.add_argument("--eval-batch-size", type=int, default=1)
    train_parser.add_argument("--eval-threshold", type=float, default=0.5)
    train_parser.add_argument("--eval-thresholds", default="0.2,0.3,0.4,0.5,0.6")
    train_parser.add_argument("--eval-max-batches", type=int, default=0)
    train_parser.add_argument("--analysis-dir", default="analysis_only")
    train_parser.add_argument("--segmentation-dir", default="segmentations")
    train_parser.add_argument("--image-extensions", default=".jpg,.jpeg,.png,.tif,.tiff,.bmp")
    train_parser.add_argument("--image-root", default="")
    train_parser.add_argument("--output-dir", default="reasoning_grounding_runs")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--max-text-len", type=int, default=128)
    train_parser.add_argument("--max-question-len", type=int, default=64)
    train_parser.add_argument("--rules-path", default="")
    train_parser.add_argument("--rag-top-k", type=int, default=5)
    train_parser.add_argument("--pieanet-model-path", default="")
    train_parser.add_argument("--backbone-type", choices=["compact", "llama"], default="compact")
    train_parser.add_argument("--llama-model-name", default="")
    train_parser.add_argument("--llama-tokenizer-name", default="")
    train_parser.add_argument("--llama-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    train_parser.add_argument("--freeze-llama", action="store_true")
    train_parser.add_argument("--use-lora", action="store_true")
    train_parser.add_argument("--lora-r", type=int, default=8)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    train_parser.add_argument("--local-files-only", action="store_true")
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--text-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--seg-loss-type", choices=["bce", "bce_dice", "bce_focal_dice"], default="bce_dice")
    train_parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--focal-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--focal-alpha", type=float, default=0.25)
    train_parser.add_argument("--focal-gamma", type=float, default=2.0)
    train_parser.add_argument("--pos-weight", type=float, default=0.0)
    train_parser.add_argument("--auto-pos-weight", action="store_true")
    train_parser.add_argument("--max-pos-weight", type=float, default=8.0)
    train_parser.add_argument("--min-freq", type=int, default=1)
    train_parser.add_argument("--max-vocab", type=int, default=12000)
    train_parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_parser.add_argument("--early-stop-patience", type=int, default=2)
    train_parser.add_argument("--device", default="")
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--question", default="")
    infer_parser.add_argument("--rules-path", default="")
    infer_parser.add_argument("--rule-context", default="")
    infer_parser.add_argument("--grounding-records-json", default="")
    infer_parser.add_argument("--rag-top-k", type=int, default=5)
    infer_parser.add_argument("--output-image", default="reasoning_grounding_output/segmented.png")
    infer_parser.add_argument("--output-text", default="reasoning_grounding_output/report.txt")
    infer_parser.add_argument("--threshold", type=float, default=0.5)
    infer_parser.add_argument("--max-text-len", type=int, default=96)
    infer_parser.add_argument("--device", default="")
    infer_parser.set_defaults(func=infer)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--test-jsonl", default="")
    eval_parser.add_argument("--dataset-format", choices=["auto", "jsonl", "dataset_example"], default="auto")
    eval_parser.add_argument("--dataset-root", default="")
    eval_parser.add_argument("--analysis-dir", default="analysis_only")
    eval_parser.add_argument("--segmentation-dir", default="segmentations")
    eval_parser.add_argument("--image-extensions", default=".jpg,.jpeg,.png,.tif,.tiff,.bmp")
    eval_parser.add_argument("--image-root", default="")
    eval_parser.add_argument("--output-json", default="reasoning_grounding_runs/eval_metrics.json")
    eval_parser.add_argument("--batch-size", type=int, default=1)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--image-size", type=int, default=0)
    eval_parser.add_argument("--max-text-len", type=int, default=0)
    eval_parser.add_argument("--max-question-len", type=int, default=0)
    eval_parser.add_argument("--rules-path", default="")
    eval_parser.add_argument("--rag-top-k", type=int, default=5)
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6")
    eval_parser.add_argument("--max-batches", type=int, default=0)
    eval_parser.add_argument("--device", default="")
    eval_parser.set_defaults(func=evaluate)
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
