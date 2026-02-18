import os
import csv
from collections import defaultdict, Counter

# ==============================================================================
# Detailed error analysis for 공격성 (aggression) and 학교 폭력 (school violence)
# ==============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "MIRROR_S1")

# Ordinal scale for aggression (공격성): 1=lowest, 4=highest
AGGRESSION_SCALE = {
    "전혀 그렇지 않다": 1,
    "그렇지 않은 편이다": 2,
    "그런 편이다": 3,
    "매우 그렇다": 4,
}

# Ordinal scale for school violence (학교 폭력)
VIOLENCE_SCALE = {
    "전혀 하지 않는다": 1,
    "거의 하지 않는다": 2,
    "가끔 한다": 3,
    "자주 한다": 4,
    "항상 한다": 5,
}

# ---- Data collection --------------------------------------------------------

# Aggression data structures
aggression_errors = []           # list of (sid, question, pred_val, gt, pred_num, reasoning)
aggression_all = []              # all aggression rows (for distribution)
aggression_error_patterns = Counter()   # "pred_val -> gt" pattern counts
aggression_too_high = 0
aggression_too_low = 0
aggression_total = 0
aggression_correct = 0

# School violence data structures
violence_errors = []
violence_all = []
violence_cold_start_false_negative = 0  # predicted "전혀 하지 않는다" but GT was something else
violence_cold_start_total = 0
violence_cold_start_errors = []
violence_total = 0
violence_correct = 0

# ---- Read all CSV files ------------------------------------------------------

for fname in sorted(os.listdir(RESULTS_DIR)):
    if not (fname.endswith(".csv") and fname.startswith("pred_")):
        continue
    filepath = os.path.join(RESULTS_DIR, fname)
    with open(filepath, encoding="utf-8") as fp:
        reader = csv.reader(fp)
        header = next(reader)
        for row in reader:
            if len(row) < 10:
                continue
            sid, question, category, pred_num, pred_val, gt, is_correct, method, setting, reasoning = row
            is_c = is_correct.strip() == "True"

            # ---- Aggression analysis ----
            if category == "공격성":
                aggression_total += 1
                aggression_all.append((sid, question, pred_val, gt, pred_num, reasoning, is_c))

                if is_c:
                    aggression_correct += 1
                else:
                    aggression_errors.append((sid, question, pred_val, gt, pred_num, reasoning))
                    pattern = f"{pred_val} -> {gt}"
                    aggression_error_patterns[pattern] += 1

                    # Determine direction of error using ordinal scale
                    pred_rank = AGGRESSION_SCALE.get(pred_val)
                    gt_rank = AGGRESSION_SCALE.get(gt)
                    if pred_rank is not None and gt_rank is not None:
                        if pred_rank > gt_rank:
                            aggression_too_high += 1
                        elif pred_rank < gt_rank:
                            aggression_too_low += 1

            # ---- School violence analysis ----
            if category == "학교 폭력":
                violence_total += 1
                violence_all.append((sid, question, pred_val, gt, pred_num, reasoning, is_c))

                if is_c:
                    violence_correct += 1
                else:
                    violence_errors.append((sid, question, pred_val, gt, pred_num, reasoning))

                # Cold-start analysis for school violence
                if "cold_start" in reasoning:
                    violence_cold_start_total += 1
                    if not is_c and pred_val == "전혀 하지 않는다":
                        violence_cold_start_false_negative += 1
                        violence_cold_start_errors.append((sid, question, pred_val, gt, reasoning))


# ==============================================================================
# SECTION 1: Aggression (공격성) Error Analysis
# ==============================================================================

print("=" * 80)
print("SECTION 1: 공격성 (Aggression) Error Analysis")
print("=" * 80)

print(f"\n  Overall: {aggression_correct}/{aggression_total} correct "
      f"({100*aggression_correct/aggression_total:.1f}% accuracy)")
print(f"  Errors:  {aggression_total - aggression_correct}")

# ---- 1a. All individual errors ----
print(f"\n--- 1a. All Aggression Errors ({len(aggression_errors)} total) ---")
print(f"  {'StudentID':<12} {'Pred_Val':<24} {'Ground_Truth':<24} {'Reasoning'}")
print(f"  {'-'*12} {'-'*24} {'-'*24} {'-'*40}")
for sid, question, pred_val, gt, pred_num, reasoning in aggression_errors:
    # Extract short question label
    q_short = question.split("-", 1)[-1].rstrip("]") if "-" in question else question
    if len(q_short) > 25:
        q_short = q_short[:25] + "..."
    print(f"  {sid:<12} {pred_val:<24} {gt:<24} {reasoning}")

# ---- 1b. Most common error patterns ----
print(f"\n--- 1b. Most Common Aggression Error Patterns ---")
print(f"  (Format: 'Predicted -> Ground Truth'  count)")
print()
for pattern, count in aggression_error_patterns.most_common():
    pct = 100 * count / len(aggression_errors) if aggression_errors else 0
    print(f"  [{count:>4}x] ({pct:>5.1f}%)  {pattern}")

# ---- 1c. Direction of errors ----
print(f"\n--- 1c. Aggression Error Direction ---")
total_directional = aggression_too_high + aggression_too_low
print(f"  Prediction TOO HIGH (overestimated aggression): {aggression_too_high}"
      f" ({100*aggression_too_high/total_directional:.1f}%)" if total_directional else "")
print(f"  Prediction TOO LOW  (underestimated aggression): {aggression_too_low}"
      f" ({100*aggression_too_low/total_directional:.1f}%)" if total_directional else "")

# ---- 1d. Distribution of predictions vs ground truths ----
print(f"\n--- 1d. Distribution: Predictions vs Ground Truths (공격성) ---")

pred_dist = Counter()
gt_dist = Counter()
for sid, question, pred_val, gt, pred_num, reasoning, is_c in aggression_all:
    pred_dist[pred_val] += 1
    gt_dist[gt] += 1

all_labels = sorted(set(list(pred_dist.keys()) + list(gt_dist.keys())),
                    key=lambda x: AGGRESSION_SCALE.get(x, 99))

print(f"\n  {'Value':<28} {'Predictions':>12} {'Ground Truths':>14} {'Diff':>8}")
print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*8}")
for label in all_labels:
    p = pred_dist.get(label, 0)
    g = gt_dist.get(label, 0)
    diff = p - g
    sign = "+" if diff > 0 else ""
    print(f"  {label:<28} {p:>12} {g:>14} {sign}{diff:>7}")

print(f"\n  Total predictions:   {sum(pred_dist.values())}")
print(f"  Total ground truths: {sum(gt_dist.values())}")

# ---- 1e. Per-question aggression accuracy ----
print(f"\n--- 1e. Per-Question Aggression Accuracy ---")
q_stats = defaultdict(lambda: {"total": 0, "correct": 0})
for sid, question, pred_val, gt, pred_num, reasoning, is_c in aggression_all:
    q_label = question.split("-", 1)[-1].rstrip("]") if "-" in question else question
    q_stats[q_label]["total"] += 1
    if is_c:
        q_stats[q_label]["correct"] += 1

print(f"\n  {'Question':<50} {'Correct':>8} {'Total':>6} {'Acc%':>7}")
print(f"  {'-'*50} {'-'*8} {'-'*6} {'-'*7}")
for q_label, stats in sorted(q_stats.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1)):
    acc = 100 * stats["correct"] / stats["total"]
    print(f"  {q_label:<50} {stats['correct']:>8} {stats['total']:>6} {acc:>6.1f}%")

# ---- 1f. Aggression errors by reasoning type ----
print(f"\n--- 1f. Aggression Errors by Reasoning Type ---")
reason_err = Counter()
reason_total = Counter()
for sid, question, pred_val, gt, pred_num, reasoning, is_c in aggression_all:
    reason_total[reasoning] += 1
    if not is_c:
        reason_err[reasoning] += 1

print(f"\n  {'Reasoning':<50} {'Errors':>7} {'Total':>7} {'ErrRate':>8}")
print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*8}")
for reason in sorted(reason_total.keys(), key=lambda r: -reason_err.get(r, 0)):
    errs = reason_err.get(reason, 0)
    tot = reason_total[reason]
    rate = 100 * errs / tot
    print(f"  {reason:<50} {errs:>7} {tot:>7} {rate:>7.1f}%")


# ==============================================================================
# SECTION 2: 학교 폭력 (School Violence) Cold-Start Error Analysis
# ==============================================================================

print("\n")
print("=" * 80)
print("SECTION 2: 학교 폭력 (School Violence) Cold-Start Error Analysis")
print("=" * 80)

print(f"\n  Overall: {violence_correct}/{violence_total} correct "
      f"({100*violence_correct/violence_total:.1f}% accuracy)")
print(f"  Errors:  {violence_total - violence_correct}")

# ---- 2a. Cold-start "전혀 하지 않는다" false predictions ----
print(f"\n--- 2a. Cold-Start Predicted '전혀 하지 않는다' but GT Was Something Else ---")
print(f"  Cold-start total rows:       {violence_cold_start_total}")
print(f"  Predicted '전혀 하지 않는다' when GT differed: {violence_cold_start_false_negative}")
if violence_cold_start_total > 0:
    print(f"  Rate: {100*violence_cold_start_false_negative/violence_cold_start_total:.1f}% "
          f"of cold-start predictions are false '전혀 하지 않는다'")

# What was the actual GT when this happened?
cold_start_gt_dist = Counter()
for sid, question, pred_val, gt, reasoning in violence_cold_start_errors:
    cold_start_gt_dist[gt] += 1

if cold_start_gt_dist:
    print(f"\n  When predicted '전혀 하지 않는다' incorrectly, actual GT was:")
    for gt_val, cnt in cold_start_gt_dist.most_common():
        print(f"    {gt_val:<28} {cnt:>5}x")

# ---- 2b. All cold-start errors (detailed list) ----
print(f"\n--- 2b. All Cold-Start '전혀 하지 않는다' Errors (sample, up to 30) ---")
print(f"  {'StudentID':<12} {'Predicted':<24} {'Ground_Truth':<24} {'Question (short)'}")
print(f"  {'-'*12} {'-'*24} {'-'*24} {'-'*40}")
for sid, question, pred_val, gt, reasoning in violence_cold_start_errors[:30]:
    q_short = question.split("-", 1)[-1].rstrip("]") if "-" in question else question
    if len(q_short) > 40:
        q_short = q_short[:40] + "..."
    print(f"  {sid:<12} {pred_val:<24} {gt:<24} {q_short}")

if len(violence_cold_start_errors) > 30:
    print(f"  ... and {len(violence_cold_start_errors) - 30} more")

# ---- 2c. Per-question cold-start error rate ----
print(f"\n--- 2c. Per-Question Cold-Start Error Breakdown ---")
cs_q_stats = defaultdict(lambda: {"total": 0, "errors": 0})
for sid, question, pred_val, gt, pred_num, reasoning, is_c in violence_all:
    if "cold_start" in reasoning:
        q_label = question.split("-", 1)[-1].rstrip("]") if "-" in question else question
        cs_q_stats[q_label]["total"] += 1
        if not is_c:
            cs_q_stats[q_label]["errors"] += 1

print(f"\n  {'Question':<65} {'Errors':>7} {'Total':>7} {'ErrRate':>8}")
print(f"  {'-'*65} {'-'*7} {'-'*7} {'-'*8}")
for q_label, stats in sorted(cs_q_stats.items(), key=lambda x: -x[1]["errors"]/max(x[1]["total"],1)):
    rate = 100 * stats["errors"] / stats["total"]
    print(f"  {q_label:<65} {stats['errors']:>7} {stats['total']:>7} {rate:>7.1f}%")

# ---- 2d. Distribution of predictions vs ground truths for school violence ----
print(f"\n--- 2d. Distribution: Predictions vs Ground Truths (학교 폭력) ---")

v_pred_dist = Counter()
v_gt_dist = Counter()
for sid, question, pred_val, gt, pred_num, reasoning, is_c in violence_all:
    v_pred_dist[pred_val] += 1
    v_gt_dist[gt] += 1

all_v_labels = sorted(set(list(v_pred_dist.keys()) + list(v_gt_dist.keys())),
                      key=lambda x: VIOLENCE_SCALE.get(x, 99))

print(f"\n  {'Value':<28} {'Predictions':>12} {'Ground Truths':>14} {'Diff':>8}")
print(f"  {'-'*28} {'-'*12} {'-'*14} {'-'*8}")
for label in all_v_labels:
    p = v_pred_dist.get(label, 0)
    g = v_gt_dist.get(label, 0)
    diff = p - g
    sign = "+" if diff > 0 else ""
    print(f"  {label:<28} {p:>12} {g:>14} {sign}{diff:>7}")

# ---- 2e. School violence error patterns ----
print(f"\n--- 2e. School Violence Error Patterns ---")
v_error_patterns = Counter()
for sid, question, pred_val, gt, pred_num, reasoning in violence_errors:
    pattern = f"{pred_val} -> {gt}"
    v_error_patterns[pattern] += 1

print(f"  (Format: 'Predicted -> Ground Truth'  count)")
print()
for pattern, count in v_error_patterns.most_common():
    total_errs = len(violence_errors)
    pct = 100 * count / total_errs if total_errs else 0
    print(f"  [{count:>4}x] ({pct:>5.1f}%)  {pattern}")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n  공격성 (Aggression):")
print(f"    Accuracy: {aggression_correct}/{aggression_total} "
      f"({100*aggression_correct/aggression_total:.1f}%)")
print(f"    Errors too high: {aggression_too_high}, too low: {aggression_too_low}")
if aggression_error_patterns:
    top_pattern, top_count = aggression_error_patterns.most_common(1)[0]
    print(f"    Most common error pattern: '{top_pattern}' ({top_count}x)")

print(f"\n  학교 폭력 (School Violence):")
print(f"    Accuracy: {violence_correct}/{violence_total} "
      f"({100*violence_correct/violence_total:.1f}%)")
print(f"    Cold-start '전혀 하지 않는다' false predictions: "
      f"{violence_cold_start_false_negative}/{violence_cold_start_total}")
if v_error_patterns:
    top_v_pattern, top_v_count = v_error_patterns.most_common(1)[0]
    print(f"    Most common error pattern: '{top_v_pattern}' ({top_v_count}x)")
