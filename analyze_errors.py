import os, csv
from collections import defaultdict

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "MIRROR_S1")

# Track by category and by reasoning type
cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})
reason_stats = defaultdict(lambda: {"total": 0, "correct": 0})
mode_stats = defaultdict(lambda: {"total": 0, "correct": 0})  # cold_start vs existing

errors_by_cat = defaultdict(list)

for f in sorted(os.listdir(d)):
    if f.endswith(".csv") and f.startswith("pred_"):
        with open(os.path.join(d, f)) as fp:
            r = csv.reader(fp)
            header = next(r)
            for row in r:
                sid, q, cat, pred_num, pred_val, gt, is_correct, method, setting, reasoning = row
                cat_stats[cat]["total"] += 1
                is_c = is_correct == "True"
                if is_c:
                    cat_stats[cat]["correct"] += 1
                
                # Parse reasoning to get mode
                parts = reasoning.split(":")
                if len(parts) >= 2:
                    mode = parts[1]  # cold_start or existing
                    mode_stats[mode]["total"] += 1
                    if is_c:
                        mode_stats[mode]["correct"] += 1
                
                reason_stats[reasoning]["total"] += 1
                if is_c:
                    reason_stats[reasoning]["correct"] += 1
                
                if not is_c:
                    errors_by_cat[cat].append((sid, q[:40], pred_val, gt, reasoning))

print("=== Accuracy by Category ===")
for cat, stats in sorted(cat_stats.items()):
    acc = 100 * stats["correct"] / stats["total"]
    print(f"  {cat}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

print("\n=== Accuracy by Mode (cold_start vs existing) ===")
for mode, stats in sorted(mode_stats.items()):
    acc = 100 * stats["correct"] / stats["total"]
    print(f"  {mode}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

print("\n=== Accuracy by Reasoning Type ===")
for reason, stats in sorted(reason_stats.items(), key=lambda x: -x[1]["total"]):
    acc = 100 * stats["correct"] / stats["total"]
    print(f"  {reason}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

print("\n=== Sample Errors by Category ===")
for cat, errors in sorted(errors_by_cat.items()):
    print(f"\n  [{cat}] ({len(errors)} errors):")
    for sid, q, pred, gt, reason in errors[:5]:
        print(f"    {sid}: {q}... pred={pred} gt={gt} ({reason})")
