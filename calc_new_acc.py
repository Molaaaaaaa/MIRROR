import os, csv
from datetime import datetime

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "MIRROR_S1")
# The new experiment started around 16:52
cutoff_time = datetime(2026, 2, 17, 16, 52, 0).timestamp()

new_total = 0
new_correct = 0
new_files = 0
old_total = 0
old_correct = 0
old_files = 0
new_accs = []
old_accs = []

for f in sorted(os.listdir(d)):
    if f.endswith(".csv") and f.startswith("pred_"):
        filepath = os.path.join(d, f)
        mtime = os.path.getmtime(filepath)
        is_new = mtime > cutoff_time
        
        s_total = 0
        s_correct = 0
        with open(filepath, encoding="utf-8") as fp:
            reader = csv.reader(fp)
            next(reader)
            for row in reader:
                s_total += 1
                if row[6] == "True":
                    s_correct += 1
        
        acc = 100 * s_correct / s_total if s_total > 0 else 0
        
        if is_new:
            new_total += s_total
            new_correct += s_correct
            new_files += 1
            new_accs.append(acc)
        else:
            old_total += s_total
            old_correct += s_correct
            old_files += 1
            old_accs.append(acc)

print(f"=== NEW RUN (after 16:52) ===")
print(f"  {new_files} students, {new_correct}/{new_total} correct = {100*new_correct/new_total:.2f}%" if new_total > 0 else "No new files yet")
if new_accs:
    print(f"  Mean per-student: {sum(new_accs)/len(new_accs):.2f}%")
    new_accs.sort()
    print(f"  Min: {min(new_accs):.0f}%, Max: {max(new_accs):.0f}%, Median: {new_accs[len(new_accs)//2]:.0f}%")

print(f"\n=== OLD RUN (before 16:52) ===")
print(f"  {old_files} students, {old_correct}/{old_total} correct = {100*old_correct/old_total:.2f}%" if old_total > 0 else "No old files")
if old_accs:
    print(f"  Mean per-student: {sum(old_accs)/len(old_accs):.2f}%")

print(f"\n=== PROJECTED TOTAL ===")
if new_total > 0:
    new_rate = new_correct / new_total
    projected_total = int(new_rate * 2500)
    print(f"  If all 100 students match new rate: {100*new_rate:.2f}% = {projected_total}/2500")
