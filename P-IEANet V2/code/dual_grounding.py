import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class GroundingRegion:
    region_id: str
    label: str
    detector_score: Optional[float]
    box: Optional[List[float]]
    mask_path: Optional[str]
    heatmap_path: Optional[str]
    average_exposure_score: Optional[float]
    min_exposure_score: Optional[float]
    max_exposure_score: Optional[float]
    exposure_std: Optional[float]
    exposure_quality: str
    rule_ids: List[str] = field(default_factory=list)


def classify_exposure(avg: Optional[float], std: Optional[float]) -> str:
    try:
        avg_value = float(avg)
    except (TypeError, ValueError):
        return "unknown"

    try:
        std_value = float(std)
    except (TypeError, ValueError):
        std_value = 0.0

    if avg_value < -0.15:
        base = "poor_or_underexposed"
    elif avg_value > 0.45:
        base = "good"
    else:
        base = "acceptable_or_mixed"

    if std_value > 0.25:
        return f"{base}_with_uneven_regions"
    return base


class DualGroundingBuilder:
    """
    Builds dual grounding evidence:
    1. visual grounding: object label, box, mask image, exposure heatmap
    2. textual grounding: retrieved rule ids and natural-language evidence lines
    """

    def build(
        self,
        segmentation_results: Dict[str, Any],
        exposure_results: Sequence[Dict[str, Any]],
        retrieved_rules: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rule_ids = [rule["rule_id"] for rule in retrieved_rules]
        regions: List[GroundingRegion] = []

        for idx, item in enumerate(exposure_results, 1):
            object_info = item.get("object_info", {})
            exposure_data = item.get("exposure_data", {})
            label = str(object_info.get("phrase") or f"object_{idx}")
            avg = exposure_data.get("average_exposure_score")
            std = exposure_data.get("exposure_std")

            regions.append(
                GroundingRegion(
                    region_id=f"region_{idx:03d}",
                    label=label,
                    detector_score=object_info.get("score"),
                    box=object_info.get("box"),
                    mask_path=object_info.get("image_path"),
                    heatmap_path=exposure_data.get("heatmap_path"),
                    average_exposure_score=avg,
                    min_exposure_score=exposure_data.get("min_exposure_score"),
                    max_exposure_score=exposure_data.get("max_exposure_score"),
                    exposure_std=std,
                    exposure_quality=classify_exposure(avg, std),
                    rule_ids=self._select_region_rules(label, avg, std, retrieved_rules, rule_ids),
                )
            )

        text_evidence = self._build_text_evidence(regions, retrieved_rules)
        manifest = {
            "text_prompt": segmentation_results.get("text_prompt", ""),
            "total_regions": len(regions),
            "regions": [asdict(region) for region in regions],
            "retrieved_rule_ids": rule_ids,
            "text_evidence": text_evidence,
        }
        return manifest

    def save_manifest(self, manifest: Dict[str, Any], output_dir: str, base_name: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{base_name}_dual_grounding.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path

    def format_for_prompt(self, manifest: Dict[str, Any]) -> str:
        lines = ["Dual-Grounding evidence:"]
        for region in manifest.get("regions", []):
            lines.append(
                f"- {region['region_id']} | label={region['label']} | "
                f"box={region.get('box')} | detector_score={region.get('detector_score')} | "
                f"avg={region.get('average_exposure_score')} | min={region.get('min_exposure_score')} | "
                f"max={region.get('max_exposure_score')} | std={region.get('exposure_std')} | "
                f"quality={region.get('exposure_quality')} | rules={','.join(region.get('rule_ids') or [])}"
            )
            lines.append(f"  mask: {region.get('mask_path')}")
            lines.append(f"  heatmap: {region.get('heatmap_path')}")

        if manifest.get("text_evidence"):
            lines.append("Textual grounding statements:")
            lines.extend([f"- {line}" for line in manifest["text_evidence"]])

        return "\n".join(lines)

    def _select_region_rules(
        self,
        label: str,
        avg: Optional[float],
        std: Optional[float],
        retrieved_rules: Sequence[Dict[str, Any]],
        fallback_rule_ids: Sequence[str],
    ) -> List[str]:
        selected: List[str] = []
        label_lower = label.lower()

        for rule in retrieved_rules:
            keywords = " ".join(rule.get("keywords", [])).lower()
            if any(token in keywords for token in label_lower.split()):
                selected.append(rule["rule_id"])

        try:
            if avg is not None and float(avg) < -0.15:
                selected.append("R001")
        except (TypeError, ValueError):
            pass

        try:
            if std is not None and float(std) > 0.25:
                selected.append("R002")
        except (TypeError, ValueError):
            pass

        selected.extend([rid for rid in fallback_rule_ids if rid in {"R004", "R006"}])
        deduped = []
        for rule_id in selected:
            if rule_id not in deduped:
                deduped.append(rule_id)
        return deduped

    def _build_text_evidence(
        self,
        regions: Sequence[GroundingRegion],
        retrieved_rules: Sequence[Dict[str, Any]],
    ) -> List[str]:
        rule_map = {rule["rule_id"]: rule["title"] for rule in retrieved_rules}
        evidence: List[str] = []
        for region in regions:
            rule_titles = [rule_map.get(rule_id, rule_id) for rule_id in region.rule_ids]
            evidence.append(
                f"{region.region_id} grounds '{region.label}' with mask '{region.mask_path}' "
                f"and heatmap '{region.heatmap_path}'. The exposure quality is "
                f"{region.exposure_quality}; avg={region.average_exposure_score}, "
                f"std={region.exposure_std}. Applicable rules: {', '.join(rule_titles) or 'none'}."
            )
        return evidence
