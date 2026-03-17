from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None


_KNOWN_CODES = ("nt", "nr", "ur", "?")
_SPLITS = ("train", "val", "test")
_KNOWN_RE = re.compile(r"\[(\?|nt|nr|ur)\]", flags=re.IGNORECASE)
_ANY_BRACKET_RE = re.compile(r"\[[^\]]+\]")


DEFAULT_TEXT_POLICY: dict[str, Any] = {
    "name": "readable_text_v1",
    "version": 1,
    "unknown_bracket_action": {
        "train": "drop_line",
        "val": "drop_line",
        "test": "drop_line",
    },
    "empty_after_transform_action": {
        "train": "drop_line",
        "val": "drop_line",
        "test": "drop_line",
    },
    "markers": {
        "nt": {"train": "remove_span", "val": "drop_line", "test": "drop_line"},
        "nr": {"train": "remove_span", "val": "drop_line", "test": "drop_line"},
        "ur": {"train": "remove_span", "val": "drop_line", "test": "drop_line"},
        "?": {"train": "drop_line", "val": "drop_line", "test": "drop_line"},
    },
}


def _normalize_split(split: str) -> str:
    split_norm = (split or "").strip().lower()
    return split_norm if split_norm in _SPLITS else "train"


def _normalize_action(value: str | None, default: str) -> str:
    action = (value or "").strip().lower()
    if action in {"keep_literal", "remove_span", "drop_line"}:
        return action
    return default


def normalize_text_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    base = json.loads(json.dumps(DEFAULT_TEXT_POLICY))
    if not policy:
        return base

    if policy.get("name"):
        base["name"] = str(policy["name"])
    if policy.get("version") is not None:
        try:
            base["version"] = int(policy["version"])
        except Exception:
            pass

    for key in ("unknown_bracket_action", "empty_after_transform_action"):
        if isinstance(policy.get(key), dict):
            for split in _SPLITS:
                base[key][split] = _normalize_action(policy[key].get(split), base[key][split])

    marker_cfg = policy.get("markers") if isinstance(policy.get("markers"), dict) else {}
    for code in _KNOWN_CODES:
        src = marker_cfg.get(code, {}) if isinstance(marker_cfg.get(code), dict) else {}
        for split in _SPLITS:
            base["markers"][code][split] = _normalize_action(src.get(split), base["markers"][code][split])

    return base


def load_text_policy_from_automation_toml(
    automation_toml: Path,
    profile_name: str | None = None,
) -> dict[str, Any]:
    if tomllib is None or not automation_toml.exists():
        return normalize_text_policy(None)

    try:
        raw = tomllib.loads(automation_toml.read_text(encoding="utf-8"))
    except Exception:
        return normalize_text_policy(None)

    text_policy_cfg = raw.get("text_policy")
    if not isinstance(text_policy_cfg, dict):
        return normalize_text_policy(None)

    profiles = text_policy_cfg.get("profiles")
    if not isinstance(profiles, dict):
        return normalize_text_policy(None)

    selected = profile_name or text_policy_cfg.get("profile") or DEFAULT_TEXT_POLICY["name"]
    selected = str(selected)
    chosen = profiles.get(selected)
    if not isinstance(chosen, dict):
        return normalize_text_policy(None)

    merged = dict(chosen)
    merged.setdefault("name", selected)
    return normalize_text_policy(merged)


def policy_hash(policy: dict[str, Any]) -> str:
    payload = json.dumps(normalize_text_policy(policy), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_text_policy_table(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS text_policy_configs (
            id            INTEGER PRIMARY KEY,
            name          TEXT NOT NULL,
            policy_hash   TEXT NOT NULL UNIQUE,
            policy_json   TEXT NOT NULL,
            source        TEXT NOT NULL DEFAULT 'automation.toml',
            notes         TEXT,
            created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_text_policy_configs_created_at
        ON text_policy_configs(created_at DESC);
        """
    )
    conn.commit()


def save_text_policy_config(
    conn,
    policy: dict[str, Any],
    source: str = "automation.toml",
    notes: str | None = None,
) -> int:
    ensure_text_policy_table(conn)
    normalized = normalize_text_policy(policy)
    p_hash = policy_hash(normalized)
    name = str(normalized.get("name") or "unnamed")
    policy_json = json.dumps(normalized, ensure_ascii=False, sort_keys=True)

    existing = conn.execute(
        "SELECT id FROM text_policy_configs WHERE policy_hash = ? LIMIT 1",
        (p_hash,),
    ).fetchone()
    if existing:
        return int(existing["id"])

    cur = conn.execute(
        """
        INSERT INTO text_policy_configs (name, policy_hash, policy_json, source, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, p_hash, policy_json, source, notes),
    )
    conn.commit()
    return int(cur.lastrowid)


@dataclass
class PolicyDecision:
    keep: bool
    action: str
    original_text: str
    text_out: str
    split: str
    marker_codes: list[str]
    has_unknown_brackets: bool


def find_marker_codes(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(1).lower() for m in _KNOWN_RE.finditer(text)]


def _has_unknown_brackets(text: str) -> bool:
    for chunk in _ANY_BRACKET_RE.findall(text or ""):
        inner = chunk[1:-1].strip().lower()
        if inner not in _KNOWN_CODES:
            return True
    return False


def _remove_marker_spans(text: str, codes: list[str]) -> str:
    out = text
    for code in set(codes):
        token = re.escape(f"[{code}]")
        out = re.sub(token, " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def apply_text_policy(text: str, split: str, policy: dict[str, Any]) -> PolicyDecision:
    split_norm = _normalize_split(split)
    normalized = normalize_text_policy(policy)
    original = text or ""
    codes = find_marker_codes(original)
    has_unknown = _has_unknown_brackets(original)

    if has_unknown:
        unknown_action = normalized["unknown_bracket_action"][split_norm]
        if unknown_action == "drop_line":
            return PolicyDecision(
                keep=False,
                action="drop_line",
                original_text=original,
                text_out="",
                split=split_norm,
                marker_codes=codes,
                has_unknown_brackets=True,
            )

    remove_codes: list[str] = []
    for code in codes:
        action = normalized["markers"][code][split_norm]
        if action == "drop_line":
            return PolicyDecision(
                keep=False,
                action="drop_line",
                original_text=original,
                text_out="",
                split=split_norm,
                marker_codes=codes,
                has_unknown_brackets=has_unknown,
            )
        if action == "remove_span":
            remove_codes.append(code)

    text_out = _remove_marker_spans(original, remove_codes) if remove_codes else original

    if not text_out.strip() and normalized["empty_after_transform_action"][split_norm] == "drop_line":
        return PolicyDecision(
            keep=False,
            action="drop_line",
            original_text=original,
            text_out="",
            split=split_norm,
            marker_codes=codes,
            has_unknown_brackets=has_unknown,
        )

    action = "keep_literal" if text_out == original else "remove_span"
    return PolicyDecision(
        keep=True,
        action=action,
        original_text=original,
        text_out=text_out,
        split=split_norm,
        marker_codes=codes,
        has_unknown_brackets=has_unknown,
    )


def summarize_text_policy_rows(rows: list[dict], policy: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "total": len(rows),
        "kept": 0,
        "dropped": 0,
        "clean_input": 0,
        "with_markers": 0,
        "transformed": 0,
        "dropped_unknown_brackets": 0,
        "marker_occurrences": {k: 0 for k in _KNOWN_CODES},
        "by_split": {},
    }

    for split in _SPLITS:
        summary["by_split"][split] = {
            "total": 0,
            "kept": 0,
            "dropped": 0,
            "clean_input": 0,
            "with_markers": 0,
            "transformed": 0,
        }

    for row in rows:
        split = _normalize_split(str(row.get("split") or "train"))
        text = str(row.get("text") or "")
        decision = apply_text_policy(text=text, split=split, policy=policy)
        codes = decision.marker_codes

        summary["by_split"][split]["total"] += 1
        if codes:
            summary["with_markers"] += 1
            summary["by_split"][split]["with_markers"] += 1
        else:
            summary["clean_input"] += 1
            summary["by_split"][split]["clean_input"] += 1
        for c in codes:
            summary["marker_occurrences"][c] += 1

        if decision.keep:
            summary["kept"] += 1
            summary["by_split"][split]["kept"] += 1
            if decision.text_out != decision.original_text:
                summary["transformed"] += 1
                summary["by_split"][split]["transformed"] += 1
        else:
            summary["dropped"] += 1
            summary["by_split"][split]["dropped"] += 1
            if decision.has_unknown_brackets:
                summary["dropped_unknown_brackets"] += 1

    return summary
