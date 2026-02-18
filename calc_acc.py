"""
Accuracy calculator for MIRROR experiments.
Computes:
- Micro accuracy (overall correct/total)
- Domain-wise (per-category) macro accuracy (paper Section 3.3, Table 6)
- Per-student accuracy distribution
"""
import os
import csv
import sys
from collections import defaultdict


def calc_accuracy(result_dir: str):
    """Calculate micro, macro (domain-wise), and per-student accuracy."""
    total = 0
    correct = 0
    files = 0
    per_student = []

    # Domain-wise (per-category) tracking
    domain_correct = defaultdict(int)
    domain_total = defaultdict(int)

    for f in sorted(os.listdir(result_dir)):
        if f.endswith(".csv") and f.startswith("pred_"):
            s_total = 0
            s_correct = 0
            with open(os.path.join(result_dir, f), encoding='utf-8-sig') as fp:
                r = csv.DictReader(fp)
                for row in r:
                    category = row.get('Category', 'Unknown')
                    is_correct = row.get('Is_Correct', 'False') == 'True'

                    total += 1
                    s_total += 1
                    domain_total[category] += 1

                    if is_correct:
                        correct += 1
                        s_correct += 1
                        domain_correct[category] += 1

            files += 1
            acc = 100 * s_correct / s_total if s_total > 0 else 0
            per_student.append((f, acc))

    if files == 0:
        print(f"No prediction files found in {result_dir}")
        return

    # === Micro accuracy ===
    micro_acc = 100 * correct / total if total > 0 else 0
    print(f"{'=' * 70}")
    print(f"Results: {result_dir}")
    print(f"{'=' * 70}")
    print(f"\n[Micro Accuracy]")
    print(f"  {files} students, {correct}/{total} correct = {micro_acc:.2f}%")

    # === Domain-wise (per-category) macro accuracy (Paper Table 6) ===
    print(f"\n[Domain-wise Accuracy (Macro)]")
    domain_accs = []
    print(f"  {'Category':<40} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 70}")
    for category in sorted(domain_total.keys()):
        d_correct = domain_correct[category]
        d_total = domain_total[category]
        d_acc = 100 * d_correct / d_total if d_total > 0 else 0
        domain_accs.append(d_acc)
        print(f"  {category:<40} {d_correct:>8} {d_total:>8} {d_acc:>9.2f}%")

    macro_acc = sum(domain_accs) / len(domain_accs) if domain_accs else 0
    print(f"  {'-' * 70}")
    print(f"  {'Macro Average':<40} {'':>8} {'':>8} {macro_acc:>9.2f}%")
    print(f"  (Number of domains: {len(domain_accs)})")

    # === Per-student accuracy distribution ===
    print(f"\n[Per-Student Accuracy Distribution]")
    accs = [a for _, a in per_student]
    accs.sort()
    print(f"  Min: {min(accs):.0f}%, Max: {max(accs):.0f}%, Median: {accs[len(accs)//2]:.0f}%")
    print(f"  Mean: {sum(accs)/len(accs):.2f}%")

    # Count by accuracy range
    ranges = [(0, 30), (30, 50), (50, 65), (65, 80), (80, 101)]
    for lo, hi in ranges:
        count = sum(1 for a in accs if lo <= a < hi)
        print(f"  {lo}-{hi-1}%: {count} students")

    return {
        'micro_accuracy': micro_acc,
        'macro_accuracy': macro_acc,
        'domain_accuracies': {cat: 100 * domain_correct[cat] / domain_total[cat]
                              for cat in domain_total},
        'per_student': per_student,
        'total': total,
        'correct': correct,
        'files': files
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "MIRROR_S1")

    calc_accuracy(result_dir)
