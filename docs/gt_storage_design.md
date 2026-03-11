# Ground Truth Storage Design Notes

Status: selected design direction for upcoming implementation

## Problem: Segmentation Stability

When segmentation algorithm or parameters change:
- Line order may change
- Line boundaries may shift
- Lines may split/merge

Current schema links `transcriptions` to `lines` by `line_id`, which breaks when segmentation reruns.

## Chosen direction

We will use **position-based GT anchoring**.

Rationale:
- preserves labels across segmentation reruns better than raw `line_id` references,
- much simpler than character-level anchoring,
- keeps the current OCR/transcription history intact.

## Planned table

```sql
CREATE TABLE ground_truth_lines (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL,
  vertical_center_y INTEGER NOT NULL,
  horizontal_left INTEGER,
  horizontal_right INTEGER,
  text TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  created_by TEXT NOT NULL,
  notes TEXT
);
```

## Matching strategy after segmentation rerun

For each GT row, search candidate lines by:
1. same `page_id`,
2. vertical center within tolerance (initially ±20 px),
3. reasonable horizontal overlap,
4. fallback manual review for ambiguous matches.

## Relationship to current implementation

Current annotation labels still live in `transcriptions` as immutable `HUMAN_CORRECTED` / `FLAGGED` rows.

That is acceptable for the current annotation phase.

The next migration step is:
- export or copy those human-approved rows into `ground_truth_lines`,
- derive anchor positions from `lines.polygon_coords`,
- keep `transcriptions` as OCR/history storage and `ground_truth_lines` as training/evaluation storage.

## Deferred alternatives

### Option A: Position-Based Matching
Store GT with coordinate anchors instead of line IDs:
- `gt_text` table with `page_id`, `vertical_center_y`, `horizontal_span`, `text`
- When segmentation changes, match lines to GT by proximity:
  - Find line whose center Y is within N pixels of GT center Y
  - Check horizontal overlap is reasonable
  - Fuzzy match if multiple candidates

### Option B: Immutable Segmentation Versions
- Mark segmentations as `immutable=1` when GT exists
- New segmentation creates new records, doesn't delete old ones
- GT always references original segmentation
- Comparison/evaluation compares across segmentation versions

### Option C: Character-Level Position Anchors
- Store GT as list of (char, bbox) pairs
- Segmentation independence: can reconstruct line-level GT from char positions
- Most flexible but complex

## Recommendation

Use **Option A** for implementation.

Option B is still a fallback if segmentation churn becomes too high, but it is not the primary path.

Key principle: **GT should survive segmentation changes** and auto-reassociate with new line boundaries where reasonable.
