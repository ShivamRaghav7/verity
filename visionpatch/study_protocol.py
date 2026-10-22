"""Finite, source-grouped selectivity study; no model inference or fitting here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import CLASSES

EDIT_WORDS = ("camera", "laptop", "watch")
FONT_FILES = {
    "train_sans": "DejaVuSans.ttf",
    "validation_serif": "DejaVuSerif.ttf",
    "final_sans_bold": "DejaVuSans-Bold.ttf",
}


def load_study_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "visionpatch-selectivity-v2":
        raise ValueError("Unknown study protocol")
    if config.get("edit_words") != list(EDIT_WORDS):
        raise ValueError("The three edit words are fixed before evaluation")
    if config.get("classes") != CLASSES:
        raise ValueError("Study candidate vocabulary differs from the fixed eight classes")
    if config.get("new_per_class") != {"validation": 4, "final": 6}:
        raise ValueError("Study split sizes differ from the declared finite protocol")
    for split in ("train", "validation", "final"):
        if not config.get("render_conditions", {}).get(split):
            raise ValueError("Every split needs declared rendering conditions")
        for condition in config["render_conditions"][split]:
            if condition.get("font_id") not in FONT_FILES or condition.get("position") not in {"top", "center", "bottom"}:
                raise ValueError("Unknown rendering condition")
    expected_conditions = {
        "train": [{"font_id": "train_sans", "position": "bottom"}],
        "validation": [{"font_id": "validation_serif", "position": "top"}],
        "final": [{"font_id": "final_sans_bold", "position": "top"},
                  {"font_id": "final_sans_bold", "position": "center"}],
    }
    if config["render_conditions"] != expected_conditions:
        raise ValueError("Font/position holds differ from the fixed protocol")
    expected_text_conditions = {split: conditions[:1] for split, conditions in expected_conditions.items()}
    if config.get("text_render_conditions") != expected_text_conditions:
        raise ValueError("Text task conditions differ from the fixed protocol")
    if config.get("support_classes") != {"camera": ["cup", "laptop"], "laptop": ["camera", "cup"], "watch": ["cup", "laptop"]}:
        raise ValueError("Four correction pairs per edit must use the predeclared support classes")
    templates = config.get("text_template_ids", {})
    seen = set()
    for split, count in (("train", 4), ("validation", 2), ("final", 3)):
        values = templates.get(split, [])
        if len(values) != count or len(set(values)) != count or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Text template counts or identifiers are invalid")
        if set(values) & seen:
            raise ValueError("A text template cannot cross study splits")
        seen.update(values)
    candidates = config.get("editing_candidates", [])
    if len(candidates) != 3:
        raise ValueError("Exactly one object-only and two protected editing candidates are declared")
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate identifiers repeat")
    for candidate in candidates:
        if any(candidate.get(key, 0) <= 0 for key in ("covariance_ridge", "fit_ridge")):
            raise ValueError("Ridge penalties must be positive")
        if candidate.get("text_retention_weight") not in {0, 1, 4}:
            raise ValueError("Unexpected text retention weight")
    if [candidate["text_retention_weight"] for candidate in candidates] != [0, 1, 4]:
        raise ValueError("The ablation and protected candidates must remain fixed")
    if config.get("projection_finetune_learning_rates") != [0.0001, 0.0005]:
        raise ValueError("Fine-tuning grid differs from the fixed two candidates")
    if config.get("projection_finetune_steps") != 25:
        raise ValueError("Fine-tuning has a fixed finite 25-update cap")
    return config


def resolve_fonts() -> dict[str, Path]:
    """Use versioned Matplotlib font assets, never machine-dependent Windows fonts.

    The repository bundles the three DejaVu assets and their license. An
    existing Matplotlib installation is a preparation-time fallback only.
    """
    directory = Path(__file__).parent / "assets" / "fonts"
    if not all((directory / filename).is_file() for filename in FONT_FILES.values()):
        import matplotlib

        directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    result = {key: directory / filename for key, filename in FONT_FILES.items()}
    if not all(path.is_file() for path in result.values()):
        raise FileNotFoundError("Required bundled DejaVu font assets are absent")
    return result


def font_records() -> dict:
    from .data import file_hash

    records = {key: {"filename": path.name, "sha256": file_hash(path)}
               for key, path in resolve_fonts().items()}
    records["license"] = {
        "name": "DejaVu Fonts license (Bitstream Vera derivative)",
        "url": "https://dejavu-fonts.github.io/License.html",
        "source": "Repository-bundled DejaVu assets copied from Matplotlib fonts/ttf",
    }
    return records


def render_study_variant(image, variant: str, *, font_path: Path,
                         position: str = "bottom", word: str = "camera"):
    """Matched geometry: text pixels are the only blank/distractor difference."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if variant not in {"clean", "blank", "distractor"}:
        raise ValueError("Unknown study image variant")
    if position not in {"top", "center", "bottom"}:
        raise ValueError("Unknown study position")
    if word not in EDIT_WORDS:
        raise ValueError("Word is outside this declared study")
    canvas = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
    y = {"bottom": 164, "top": 12, "center": 89}[position]
    box = (32, y, 192, y + 46)
    if variant != "clean":
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(box, fill="white", outline="gray", width=1)
        if variant == "distractor":
            font = ImageFont.truetype(str(font_path), 32)
            draw.text((112, y + 23), word, font=font, fill="black", anchor="mm")
    return canvas, box


def render_text_card(word: str, template_id: int, *, font_path: Path, position: str = "bottom"):
    """Controlled legitimate text task; not a natural-scene reading dataset.

    A template's colors and line geometry are deterministic. All eight word
    counterfactuals inherit that template's split. No generated card is counted
    as a photograph or an independent human annotation.
    """
    from PIL import Image, ImageDraw, ImageFont

    if word not in CLASSES.values() or type(template_id) is not int or template_id < 0:
        raise ValueError("Unknown text-card word or template")
    if position not in {"top", "center", "bottom"}:
        raise ValueError("Unknown study position")
    seed = hashlib.sha256(f"visionpatch-text-template-{template_id}".encode()).digest()
    background = tuple(180 + component % 70 for component in seed[:3])
    card = Image.new("RGB", (224, 224), background)
    draw = ImageDraw.Draw(card)
    for index in range(5):
        coordinate = 8 + seed[3 + index] % 208
        draw.line((0, coordinate, 223, coordinate), fill=tuple(max(0, c - 25) for c in background), width=1)
    y = {"bottom": 158, "top": 12, "center": 86}[position]
    draw.rounded_rectangle((8, y, 216, y + 52), radius=4, fill="white", outline="gray")
    size = 30
    font = ImageFont.truetype(str(font_path), size)
    while draw.textbbox((0, 0), word, font=font)[2] > 196 and size > 12:
        size -= 1
        font = ImageFont.truetype(str(font_path), size)
    draw.text((112, y + 26), word, font=font, fill="black", anchor="mm")
    return card


def iter_photo_tasks(manifest: dict, split: str, config: dict) -> list[dict]:
    """Return deterministic task metadata; all variants retain source grouping.

    Clean appears once per source. Every edit/condition has a matched blank and
    distractor. The correct label always remains the source object's class.
    """
    if split not in {"train", "validation", "final"}:
        raise ValueError("Unknown study split")
    rows = []
    for source in manifest["images"]:
        if source["split"] != split:
            continue
        clean_condition = config["render_conditions"][split][0]
        rows.append({**source, "task_id": f"{source['image_id']}|clean", "task_kind": "object",
                     "variant": "clean", "edit_id": None, "word": None, **clean_condition})
        for word in EDIT_WORDS:
            for condition in config["render_conditions"][split]:
                suffix = f"{word}|{condition['font_id']}|{condition['position']}"
                for variant in ("blank", "distractor"):
                    rows.append({**source, "task_id": f"{source['image_id']}|{variant}|{suffix}",
                                 "task_kind": "object", "variant": variant,
                                 "edit_id": word, "word": word, **condition})
    return rows


def text_task_rows(split: str, config: dict) -> list[dict]:
    if split not in {"train", "validation", "final"}:
        raise ValueError("Unknown study split")
    rows = []
    for template in config["text_template_ids"][split]:
        for condition in config["text_render_conditions"][split]:
            for class_id, word in CLASSES.items():
                rows.append({"task_id": f"text-{template}|{class_id}|{condition['font_id']}|{condition['position']}",
                             "template_id": template, "split": split, "task_kind": "text",
                             "class_id": class_id, "label": word, "word": word, **condition})
    return rows
