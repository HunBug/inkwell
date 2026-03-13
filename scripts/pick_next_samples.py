#!/usr/bin/env python3
"""
Suggest the next N lines to annotate to maximise model improvement.

Strategy (default: combined):
  1. error-chars  — lines whose OCR_AUTO text contains chars the current model
                    gets wrong most, derived from a completed eval result.json
  2. diversity    — spread picks evenly across notebooks/pages; avoid clustering
  3. combined     — weighted sum of both (default)

Usage:
    # Use the latest fine-tuned eval result automatically:
    python scripts/pick_next_samples.py --n 150

    # Point at a specific eval result:
    python scripts/pick_next_samples.py --n 150 \
        --eval-result /path/to/jobs/eval_20260312_145141/result.json

    # Dry run: print summary only, don't write output file:
    python scripts/pick_next_samples.py --n 150 --dry-run

Outputs:
    working/suggestions/next_samples_{timestamp}.jsonl
        {"line_id": N, "page_id": N, "notebook": "...", "score": 0.82,
         "reasons": ["hard_chars:á,é,ő", "rare_notebook:nb03"],
         "ocr_preview": "...first 80 chars of OCR_AUTO..."}

    Prints a short summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import get_connection, DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Levenshtein alignment  (reused from eval_model.py, no extra deps)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[lb]

def _edit_ops(a: str, b: str) -> list[tuple[str, str | None, str | None]]:
    """
    Return aligned (op, ref_char, pred_char) list.
    op ∈ {'match', 'sub', 'del', 'ins'}
    
    'del'  = char in ref missing from pred  (model dropped it)
    'ins'  = char in pred not in ref        (model hallucinated)
    'sub'  = ref char replaced by pred char
    """
    la, lb = len(a), len(b)
    # Build DP table
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Backtrack
    ops = []
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(("match", a[i - 1], b[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _extract_hard_chars(per_line: list[dict], top_n: int = 40) -> dict[str, float]:
    """
    From eval per_line results, count how often each reference char was involved
    in an error (sub or del).  Returns {char: error_rate} for top_n chars.
    """
    char_errors: Counter = Counter()
    char_total: Counter = Counter()

    for item in per_line:
        ref = item.get("reference", "")
        pred = item.get("prediction", "")
        if not ref:
            continue
        ops = _edit_ops(ref, pred)
        for op, rc, _ in ops:
            if rc is None:
                continue  # insertion, no ref char
            rc_lower = rc.lower()
            char_total[rc_lower] += 1
            if op in ("sub", "del"):
                char_errors[rc_lower] += 1

    # Compute error rate; only keep chars that appeared enough times
    rates: dict[str, float] = {}
    for ch, total in char_total.items():
        if total >= 2:
            rates[ch] = char_errors.get(ch, 0) / total

    # Return top_n by error rate
    return dict(sorted(rates.items(), key=lambda kv: -kv[1])[:top_n])


def _norm_text(s: str | None) -> str:
    return " ".join((s or "").strip().lower().split())


def _char_distance_ratio(a: str, b: str) -> float:
    a2 = _norm_text(a)
    b2 = _norm_text(b)
    if not a2 and not b2:
        return 0.0
    return _edit_distance(a2, b2) / max(len(a2), len(b2), 1)


def _letter_ratio(s: str | None) -> float:
    t = _norm_text(s)
    if not t:
        return 0.0
    letters = sum(1 for ch in t if ch.isalpha())
    return letters / max(len(t), 1)


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _load_unannotated_lines(conn) -> list[dict]:
    """
    Load all unannotated lines that have at least one OCR_AUTO transcription.
    Returns list of dicts with: line_id, page_id, notebook_id, notebook_label,
                                ocr_text, ocr_conf
    """
    rows = conn.execute("""
        SELECT
            l.id           AS line_id,
            l.page_id,
            nb.id          AS notebook_id,
            nb.label       AS notebook_label,
            nb.folder_name AS notebook_folder,
            t.text         AS ocr_text,
            t.confidence   AS ocr_conf
        FROM lines l
        JOIN pages p       ON p.id = l.page_id
        JOIN source_images si ON si.id = p.source_image_id
        JOIN assets a      ON a.id = si.asset_id
        JOIN notebooks nb  ON nb.id = a.notebook_id
        -- Best single OCR_AUTO transcription per line (highest confidence)
        JOIN transcriptions t ON t.id = (
            SELECT t2.id FROM transcriptions t2
            WHERE t2.line_id = l.id
              AND t2.transcription_type = 'OCR_AUTO'
            ORDER BY t2.confidence DESC NULLS LAST, t2.id DESC
            LIMIT 1
        )
        WHERE l.skip = 0
          AND l.crop_image_path IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM transcriptions hc
              WHERE hc.line_id = l.id
                AND hc.transcription_type = 'HUMAN_CORRECTED'
                AND hc.immutable = 1
          )
        ORDER BY l.id
    """).fetchall()
    return [dict(r) for r in rows]


def _load_already_annotated_line_ids(conn) -> set[int]:
    rows = conn.execute("""
        SELECT DISTINCT line_id FROM transcriptions
        WHERE transcription_type='HUMAN_CORRECTED' AND immutable=1
    """).fetchall()
    return {r["line_id"] for r in rows}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_lines(
    candidates: list[dict],
    hard_chars: dict[str, float],
    pool_predictions: dict[int, str] | None,
    n: int,
    max_per_page: int,
    max_per_notebook: int,
) -> list[dict]:
    """
    Score and rank candidates, then apply diversity cap.
    Returns top-n items with scores and reason tags.
    """
    scored = []
    for c in candidates:
        text = (c["ocr_text"] or "").strip()
        if len(text) < 4:
            continue  # skip near-empty

        # Skip lines that look like pure noise / figure labels:
        #   - all caps with no spaces and < 8 chars (e.g. "ANOS", "PLAIN")
        #   - purely numeric (page numbers, dates as standalone)
        stripped_nospace = text.replace(" ", "").replace("-", "")
        if stripped_nospace.isupper() and len(stripped_nospace) <= 8:
            continue
        if stripped_nospace.replace(".", "").isnumeric():
            continue

        text_lower = text.lower()

        # --- Error-char score: sum of hard_char error-rates in this line ---
        ec_score = 0.0
        matched_hard: list[str] = []
        for ch, rate in hard_chars.items():
            count = text_lower.count(ch)
            if count:
                ec_score += rate * min(count, 3)  # cap per-char contribution
                matched_hard.append(ch)

        # Normalise by text length so short lines don't dominate
        ec_score = ec_score / max(len(text_lower), 1) * 10

        # --- Length score: strongly prefer medium lines (15-80 chars) ---
        tlen = len(text.strip())
        if tlen < 8:
            length_score = 0.1   # near-useless for training
        elif tlen < 15:
            length_score = 0.5   # short but ok
        elif tlen <= 80:
            length_score = 1.0   # ideal
        else:
            length_score = 0.7   # long but still useful

        # --- OCR confidence as proxy for difficulty (lower conf → harder) ---
        conf = c["ocr_conf"] or 0.5
        diff_score = max(0.0, 1.0 - conf)  # higher when model was unsure

        # --- Pool disagreement score: OCR_AUTO vs fine-tuned full-pool prediction ---
        pool_disagree = 0.0
        pool_pred_text = None
        if pool_predictions is not None:
            pred = pool_predictions.get(c["line_id"])
            if pred is not None:
                # Skip likely garbage lines where both OCR and model output are
                # mostly symbols/digits/punctuation.
                if max(_letter_ratio(c.get("ocr_text", "")), _letter_ratio(pred)) < 0.45:
                    continue
                pool_disagree = _char_distance_ratio(c.get("ocr_text", ""), pred)
                pool_pred_text = pred

        # In pool mode we trust disagreement most; without pool predictions
        # fall back to previous eval+confidence heuristic.
        if pool_predictions is not None:
            combined = pool_disagree * 0.60 + diff_score * 0.25 + length_score * 0.15
        else:
            combined = ec_score * 0.60 + diff_score * 0.25 + length_score * 0.15

        reasons: list[str] = []
        if matched_hard:
            reasons.append("hard_chars:" + ",".join(sorted(matched_hard)[:6]))
        if conf < 0.3:
            reasons.append("low_ocr_conf")
        if pool_disagree >= 0.4:
            reasons.append("pool_disagree_high")
        elif pool_disagree >= 0.2:
            reasons.append("pool_disagree_mid")
        if tlen < 15:
            reasons.append("short_line")
        elif tlen > 80:
            reasons.append("long_line")

        scored.append({
            **c,
            "score": round(combined, 4),
            "reasons": reasons,
            "ec_score": ec_score,
            "diff_score": diff_score,
            "pool_disagree": round(pool_disagree, 4),
            "pool_pred_text": pool_pred_text,
        })

    # Sort by combined score descending
    scored.sort(key=lambda x: -x["score"])

    # --- Diversity cap: max N lines per page, M per notebook ---
    page_counts: Counter = Counter()
    nb_counts: Counter = Counter()
    selected = []
    for item in scored:
        pid = item["page_id"]
        nid = item["notebook_id"]
        if page_counts[pid] >= max_per_page:
            continue
        if nb_counts[nid] >= max_per_notebook:
            continue
        selected.append(item)
        page_counts[pid] += 1
        nb_counts[nid] += 1
        if len(selected) >= n:
            break

    return selected


# ---------------------------------------------------------------------------
# Shared-folder helpers
# ---------------------------------------------------------------------------

def _find_latest_finetuned_eval(shared: Path | None) -> Path | None:
    """Return the result.json of the most recent completed fine-tuned eval job."""
    if shared is None or not shared.exists():
        return None
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return None
    best: tuple[str, Path] | None = None
    for d in sorted(jobs_dir.iterdir()):
        if not d.name.startswith("eval_"):
            continue
        r = d / "result.json"
        j = d / "job.json"
        if not r.exists() or not j.exists():
            continue
        try:
            job = json.loads(j.read_text())
            cp = job.get("eval_checkpoint", "") or ""
            # skip baseline; only fine-tuned checkpoints are local paths
            if "microsoft" in cp or not cp:
                continue
            created = job.get("created_at", "")
            if best is None or created > best[0]:
                best = (created, r)
        except Exception:
            pass
    return best[1] if best else None


def _find_baseline_eval(shared: Path | None) -> Path | None:
    """Return result.json of the most recent baseline eval."""
    if shared is None or not shared.exists():
        return None
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return None
    best: tuple[str, Path] | None = None
    for d in sorted(jobs_dir.iterdir()):
        if not d.name.startswith("eval_"):
            continue
        r = d / "result.json"
        j = d / "job.json"
        if not r.exists() or not j.exists():
            continue
        try:
            job = json.loads(j.read_text())
            cp = job.get("eval_checkpoint", "") or ""
            if "microsoft" not in cp:
                continue
            created = job.get("created_at", "")
            if best is None or created > best[0]:
                best = (created, r)
        except Exception:
            pass
    return best[1] if best else None


def _find_latest_infer_pool_predictions(shared: Path | None) -> Path | None:
    """Return pool_predictions.jsonl from latest completed infer_pool job."""
    if shared is None or not shared.exists():
        return None
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return None
    best: tuple[str, Path] | None = None
    for d in sorted(jobs_dir.iterdir()):
        if not d.name.startswith("infer_pool_"):
            continue
        r = d / "result.json"
        p = d / "pool_predictions.jsonl"
        if not r.exists() or not p.exists():
            continue
        try:
            result = json.loads(r.read_text())
        except Exception:
            continue
        if result.get("status") != "completed":
            continue
        finished = result.get("finished_at", "")
        if best is None or finished > best[0]:
            best = (finished, p)
    return best[1] if best else None


def _load_pool_predictions(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                line_id = row.get("line_id")
                pred = row.get("predicted_text")
                if isinstance(line_id, int) and isinstance(pred, str):
                    out[line_id] = pred
            except Exception:
                continue
    return out


def _get_shared(override: str | None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("INKWELL_SHARED")
    if env:
        return Path(env).expanduser().resolve()
    default = Path("/home/akoss/mnt/lara-playground/playground/inkwell-automation")
    return default if default.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest next lines to annotate")
    parser.add_argument("--n", type=int, default=150, help="Number of lines to suggest (default: 150)")
    parser.add_argument(
        "--eval-result",
        default=None,
        help="Path to eval result.json for error-char scoring. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--baseline-eval",
        default=None,
        help="Path to baseline eval result.json (for comparison in summary).",
    )
    parser.add_argument(
        "--pool-predictions",
        default=None,
        help="Path to infer_pool pool_predictions.jsonl (auto-detected if omitted).",
    )
    parser.add_argument("--max-per-page", type=int, default=2, help="Max lines picked per page (default: 2)")
    parser.add_argument("--max-per-notebook", type=int, default=40, help="Max lines picked per notebook (default: 40)")
    parser.add_argument("--shared", default=None, help="Shared folder path (overrides INKWELL_SHARED)")
    parser.add_argument("--db", default=None, help="Override DB path")
    parser.add_argument("--dry-run", action="store_true", help="Print summary only, don't write output file")
    parser.add_argument("--output-dir", default=None, help="Directory for output file (default: working/suggestions/)")
    args = parser.parse_args()

    shared = _get_shared(args.shared)
    db_path = Path(args.db) if args.db else None

    # --- Load eval result for error-char analysis ---
    eval_result_path = Path(args.eval_result) if args.eval_result else _find_latest_finetuned_eval(shared)
    baseline_result_path = Path(args.baseline_eval) if args.baseline_eval else _find_baseline_eval(shared)
    pool_predictions_path = Path(args.pool_predictions) if args.pool_predictions else _find_latest_infer_pool_predictions(shared)

    hard_chars: dict[str, float] = {}
    eval_source = "none"
    if eval_result_path and eval_result_path.exists():
        data = json.loads(eval_result_path.read_text())
        per_line = data.get("per_line", [])
        hard_chars = _extract_hard_chars(per_line, top_n=40)
        eval_source = str(eval_result_path)
        print(f"[pick] Using eval result: {eval_result_path}")
        print(f"[pick] Hard chars ({len(hard_chars)}): " +
              "  ".join(f"{ch}={v:.2f}" for ch, v in list(hard_chars.items())[:15]))
    else:
        print("[pick] No fine-tuned eval result found; scoring by OCR confidence only")

    pool_predictions: dict[int, str] | None = None
    if pool_predictions_path and pool_predictions_path.exists():
        pool_predictions = _load_pool_predictions(pool_predictions_path)
        print(f"[pick] Using full-pool predictions: {pool_predictions_path}")
        print(f"[pick] Pool prediction rows: {len(pool_predictions)}")
    else:
        print("[pick] No infer_pool predictions found; using eval-only heuristic")

    # --- Load DB candidates ---
    conn = get_connection(db_path)
    print("[pick] Loading unannotated lines from DB...")
    candidates = _load_unannotated_lines(conn)
    conn.close()
    print(f"[pick] {len(candidates)} unannotated lines with OCR_AUTO text")

    # --- Score + pick ---
    selected = _score_lines(
        candidates,
        hard_chars,
        pool_predictions,
        n=args.n,
        max_per_page=args.max_per_page,
        max_per_notebook=args.max_per_notebook,
    )
    print(f"\n[pick] Selected {len(selected)} lines")

    # --- Summary ---
    nb_dist: Counter = Counter(s["notebook_folder"] for s in selected)
    reason_dist: Counter = Counter()
    for s in selected:
        for r in s["reasons"]:
            reason_dist[r.split(":")[0]] += 1

    print("\nNotebook distribution:")
    for nb, count in sorted(nb_dist.items(), key=lambda x: -x[1]):
        print(f"  {nb}: {count}")

    print("\nReason tags:")
    for r, count in sorted(reason_dist.items(), key=lambda x: -x[1]):
        print(f"  {r}: {count}")

    if baseline_result_path and baseline_result_path.exists():
        b = json.loads(baseline_result_path.read_text())
        if eval_result_path and eval_result_path.exists():
            e = json.loads(eval_result_path.read_text())
            delta = e.get("cer", 0) - b.get("cer", 0)
            sign = "↓" if delta < 0 else "↑"
            print(f"\nModel progress: CER {b['cer']:.4f} → {e['cer']:.4f}  {sign}{abs(delta)*100:.1f}pp")

    print(f"\nTop 10 suggestions:")
    for item in selected[:10]:
        preview = (item["ocr_text"] or "")[:70]
        print(f"  [{item['score']:.3f}] line {item['line_id']} (p{item['page_id']} {item['notebook_folder']}) {preview!r}")
        if item["reasons"]:
            print(f"          → {', '.join(item['reasons'])}")

    if args.dry_run:
        print("\n[dry-run] Not writing output file.")
        return

    # --- Write output ---
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "working" / "suggestions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"next_samples_{ts}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps({
                "line_id":         item["line_id"],
                "page_id":         item["page_id"],
                "notebook_folder": item["notebook_folder"],
                "notebook_label":  item["notebook_label"],
                "score":           item["score"],
                "reasons":         item["reasons"],
                "ocr_preview":     (item["ocr_text"] or "")[:120],
                "pool_pred_preview": (item.get("pool_pred_text") or "")[:120],
            }, ensure_ascii=False) + "\n")

    print(f"\n[pick] Written: {out_path}")
    print("       Open the /annotate page to start annotating.")
    print(f"       Suggested line IDs: {selected[0]['line_id']} … {selected[-1]['line_id']}")


if __name__ == "__main__":
    main()
