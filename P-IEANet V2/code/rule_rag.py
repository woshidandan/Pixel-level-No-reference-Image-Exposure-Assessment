import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class ExposureRule:
    rule_id: str
    title: str
    condition: str
    guidance: str
    keywords: List[str]
    priority: int = 1


DEFAULT_RULES = [
    ExposureRule(
        rule_id="R001",
        title="Low exposure confidence",
        condition="The target region has a low average exposure score.",
        guidance=(
            "Treat the region as likely underexposed or poorly exposed. Mention the "
            "affected object, avoid making a whole-image claim unless most regions agree, "
            "and recommend local brightening before global exposure changes."
        ),
        keywords=["underexposed", "dark", "low", "shadow", "poor", "subject"],
        priority=3,
    ),
    ExposureRule(
        rule_id="R002",
        title="High local exposure variance",
        condition="The target region has high exposure standard deviation or a wide score range.",
        guidance=(
            "Explain that exposure is spatially uneven. Ground the explanation in the "
            "segmented region and heatmap, then recommend highlight/shadow balancing."
        ),
        keywords=["uneven", "variance", "std", "range", "mixed", "highlight", "shadow"],
        priority=3,
    ),
    ExposureRule(
        rule_id="R003",
        title="Subject-first assessment",
        condition="The user asks about a person, face, product, car, or another main subject.",
        guidance=(
            "Prioritize the subject region over the background. If the background is "
            "well exposed but the subject is not, report the subject issue as the main finding."
        ),
        keywords=["person", "face", "subject", "portrait", "car", "product", "foreground"],
        priority=2,
    ),
    ExposureRule(
        rule_id="R004",
        title="Transparent or black background mask",
        condition="The segmented image uses transparent or black pixels outside the mask.",
        guidance=(
            "Ignore transparent or black background pixels when interpreting the exposure "
            "score. Use the segmented object and its heatmap as the grounded evidence."
        ),
        keywords=["mask", "segmented", "transparent", "background", "object"],
        priority=2,
    ),
    ExposureRule(
        rule_id="R005",
        title="Good but not uniform exposure",
        condition="The average score is acceptable but min/max values show localized defects.",
        guidance=(
            "Give a balanced assessment: overall exposure can be usable while localized "
            "areas still need correction. Name the localized risk instead of only giving a score."
        ),
        keywords=["acceptable", "overall", "local", "min", "max", "usable"],
        priority=1,
    ),
    ExposureRule(
        rule_id="R006",
        title="Question-answer alignment",
        condition="The user asks a specific exposure question.",
        guidance=(
            "Answer the user's question directly first, then cite the grounded region, "
            "the exposure score, the heatmap evidence, and finally give an actionable suggestion."
        ),
        keywords=["analyze", "evaluate", "compare", "which", "why", "recommendation"],
        priority=2,
    ),
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _quality_terms(record: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    avg = record.get("average_exposure_score")
    std = record.get("exposure_std")
    min_score = record.get("min_exposure_score")
    max_score = record.get("max_exposure_score")

    try:
        if avg is not None and float(avg) < -0.15:
            terms.extend(["low", "underexposed", "dark", "poor"])
        elif avg is not None and float(avg) > 0.45:
            terms.extend(["good", "bright", "acceptable"])
        else:
            terms.extend(["acceptable", "mixed"])
    except (TypeError, ValueError):
        pass

    try:
        if std is not None and float(std) > 0.25:
            terms.extend(["uneven", "variance", "std", "mixed"])
    except (TypeError, ValueError):
        pass

    try:
        if min_score is not None and max_score is not None and float(max_score) - float(min_score) > 0.7:
            terms.extend(["range", "local", "highlight", "shadow"])
    except (TypeError, ValueError):
        pass

    return terms


class RuleRAGRetriever:
    """
    Lightweight Rule RAG for image exposure assessment.

    The retriever is intentionally dependency-free: it uses lexical overlap plus
    exposure-statistics triggers. Rules can be replaced with a JSON file without
    changing the workflow code.
    """

    def __init__(self, rules_path: Optional[str] = None, rules: Optional[Sequence[ExposureRule]] = None):
        self.rules = list(rules or DEFAULT_RULES)
        if rules_path:
            self.rules = self._load_rules(rules_path)

    def _load_rules(self, rules_path: str) -> List[ExposureRule]:
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rule file not found: {rules_path}")
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ExposureRule(**item) for item in data]

    def save_rules(self, rules_path: str) -> None:
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump([asdict(rule) for rule in self.rules], f, ensure_ascii=False, indent=2)

    def retrieve(
        self,
        user_query: str,
        grounding_records: Optional[Iterable[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        records = list(grounding_records or [])
        context_terms = _tokenize(user_query)
        for record in records:
            context_terms.extend(_tokenize(str(record.get("label", ""))))
            context_terms.extend(_quality_terms(record))

        context = set(context_terms)
        ranked: List[Dict[str, Any]] = []
        for rule in self.rules:
            rule_terms = set(_tokenize(" ".join(rule.keywords + [rule.title, rule.condition])))
            overlap = context.intersection(rule_terms)
            score = len(overlap) + 0.25 * rule.priority
            if overlap or rule.rule_id in {"R004", "R006"}:
                ranked.append({
                    "rule_id": rule.rule_id,
                    "title": rule.title,
                    "condition": rule.condition,
                    "guidance": rule.guidance,
                    "keywords": rule.keywords,
                    "priority": rule.priority,
                    "retrieval_score": round(score, 4),
                    "matched_terms": sorted(overlap),
                })

        ranked.sort(key=lambda item: (item["retrieval_score"], item["priority"]), reverse=True)
        return ranked[:top_k]

    @staticmethod
    def format_rules_for_prompt(rules: Sequence[Dict[str, Any]]) -> str:
        if not rules:
            return "No exposure rules were retrieved."

        lines = ["Retrieved Rule RAG evidence:"]
        for idx, rule in enumerate(rules, 1):
            lines.append(
                f"{idx}. [{rule['rule_id']}] {rule['title']}\n"
                f"   Condition: {rule['condition']}\n"
                f"   Guidance: {rule['guidance']}\n"
                f"   Matched terms: {', '.join(rule.get('matched_terms') or ['general'])}"
            )
        return "\n".join(lines)
