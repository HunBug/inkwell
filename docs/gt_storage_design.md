# Ground Truth Storage Design Notes

## Problem: Segmentation Stability

When segmentation algorithm or parameters change:
- Line order may change
- Line boundaries may shift
- Lines may split/merge

Current schema links `transcriptions` to `lines` by `line_id`, which breaks when segmentation reruns.

## Solution Approaches

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

## Recommendation (TBD)
Likely **Option A** (position-based) for MVP, with Option B as fallback for training data preservation.

Key principle: **GT should survive segmentation changes** and auto-reassociate with new line boundaries where reasonable.
