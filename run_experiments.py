"""
MIRROR Experiment Runner
Paper: Multi-view Inference via Retrospective Retrieval for Ontological Representation of Persona

Supported Methods:
- 2018_only: Use 2018 data only
- 2022_only: Use 2022 data only
- RER: + Retrospective Evidence Retrieval
- RER_LTE: + Longitudinal Trend Extraction
- RER_KG: + Knowledge Graph Constraints
- MIRROR: RER + LTE + KG (Full Framework)

Experimental Settings:
- S1 (Full-history): Use all 2018-2022 survey responses
- S2 (Violence-blinded): Exclude Aggression, Delinquency, School Violence
- S3 (S2 + Aggression): S2 with Aggression history restored
"""
import os
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from langchain_ollama import ChatOllama

from config import Config
from mirror_framework import create_predictor
from utils import (
    load_student_data,
    load_ground_truth,
    load_target_questions,
    filter_target_questions,
    get_all_student_ids,
    clean_llm_output,
)


SUPPORTED_METHODS = [
    '2018_only',      # Baseline: single year
    '2022_only',      # Baseline: single year
    'LLM_only',       # Baseline: no framework
    'RER',            # Evidence Retrieval
    'RER_LTE',        # Longitudinal Trends
    'RER_KG',         # Knowledge Graph
    'MIRROR',         # Full: RER + LTE + KG
]


class BaselinePredictor:
    """Baseline Predictor for single-year and LLM-only methods"""
    
    def __init__(self, student_id: str, method: str, 
                 exclude_target: bool = False, exclude_partial: bool = False):
        self.student_id = student_id
        self.method = method
        
        self.input_vars, self.history = load_student_data(
            Config.DATA_DIR, student_id, 
            exclude_target=exclude_target, 
            exclude_partial=exclude_partial
        )
        
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=50,
            num_ctx=Config.NUM_CTX,
            # reasoning=False,  # set True for reasoning models
            timeout=Config.LLM_TIMEOUT
        )
    
    def _build_history_context(self, question: str) -> str:
        """Build context from years 2018-2022 ONLY (exclude target year 2023)"""
        years = sorted(self.history.keys())
        history_lines = []
        
        for year in years:
            # CRITICAL: Exclude target year (2023) to prevent data leakage
            if int(year) >= Config.TARGET_YEAR:
                continue
            
            year_data = self.history.get(year, {})
            answer = year_data.get(question, None)
            if answer:
                history_lines.append(f"{year}년: {answer}")
        
        if history_lines:
            return "\n".join(history_lines)
        return "과거 응답 기록 없음"
    
    def _build_static_persona(self) -> str:
        """Build static persona from demographic info"""
        persona_parts = []
        if self.input_vars.get("gender"):
            persona_parts.append(f"성별: {self.input_vars['gender']}")
        if self.input_vars.get("birth_year"):
            persona_parts.append(f"출생년도: {self.input_vars['birth_year']}")
        if self.input_vars.get("region"):
            persona_parts.append(f"지역: {self.input_vars['region']}")
        
        if persona_parts:
            return " | ".join(persona_parts)
        return ""
    
    def predict(self, question: str, options: Dict[str, str]) -> Tuple[str, str]:
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        if self.method == "2018_only":
            answer = self.history.get("2018", {}).get(question, "N/A")
            context = f"2018년 응답: {answer}"
        elif self.method == "2022_only":
            answer = self.history.get("2022", {}).get(question, "N/A")
            context = f"2022년 응답: {answer}"
        elif self.method == "LLM_only":
            # Pure baseline: raw history concatenation ONLY (no static persona)
            # Static persona is LTE component - must be excluded for fair comparison
            history_context = self._build_history_context(question)
            
            context = f"[과거 응답 기록 (2018-2022)]\n{history_context}\n\n위 정보를 바탕으로 이 학생의 2023년 응답을 예측하세요."
        else:
            context = ""
        
        prompt = f"""첫 번째 줄에 답변 번호만 출력하세요.
예시:
Answer: 2

{context}

질문: {question}

선택지:
{options_str}

숫자만 출력하세요.
Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            pred = clean_llm_output(response.content)
            return pred, self.method
        except Exception as e:
            return "0", f"Error:{str(e)[:20]}"
    
    def predict_batch(self, tasks: List[Dict]) -> Tuple[List[str], List[str]]:
        preds, reasons = [], []
        desc = f"[{self.method}] {self.student_id}"
        for task in tqdm(tasks, desc=desc, leave=False):
            pred, reason = self.predict(task['question'], task['options'])
            preds.append(pred)
            reasons.append(reason)
        return preds, reasons


def run_single_method(student_id: str, method: str, targets: List[Dict], 
                      gt_map: Dict, rebuild: bool = False,
                      exclude_target: bool = False, exclude_partial: bool = False,
                      result_suffix: str = "", output_dir: str = None,
                      debug: bool = False) -> Dict:
    """Run prediction for a single student with a specific method"""
    
    base_dir = output_dir if output_dir else Config.RESULTS_DIR
    dir_suffix = result_suffix if result_suffix else ""
    result_dir = os.path.join(base_dir, f"{method}{dir_suffix}")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"pred_{student_id}.csv")
    
    # Check cache
    if os.path.exists(result_file) and not rebuild:
        try:
            df = pd.read_csv(result_file)
            correct = df['Is_Correct'].sum()
            total = len(df[df['Ground_Truth'] != 'N/A'])
            return {
                'accuracy': (correct / total * 100) if total > 0 else 0,
                'correct': correct, 'total': total, 'cached': True
            }
        except:
            pass
    
    # Select predictor based on method
    if method in ["2018_only", "2022_only", "LLM_only"]:
        predictor = BaselinePredictor(
            student_id, method, 
            exclude_target=exclude_target, 
            exclude_partial=exclude_partial
        )
        predictions, reasons = predictor.predict_batch(targets)
    
    elif method in ["RER", "RER_LTE", "RER_KG", "MIRROR"]:
        # Use create_predictor with method parameter
        predictor = create_predictor(
            student_id=student_id,
            method=method,
            exclude_target=exclude_target,
            exclude_partial=exclude_partial,
            debug=debug
        )
        # Pre-compute all question embeddings at once (batch)
        predictor.precompute_embeddings([t['question'] for t in targets])
        predictions, reasons = predictor.predict_batch(targets)
    
    else:
        print(f"[Warning] Unknown method: {method}")
        return {'accuracy': 0, 'correct': 0, 'total': 0, 'cached': False}
    
    # Evaluation
    rows = []
    correct = 0
    valid = 0
    
    for i, task in enumerate(targets):
        pred = str(predictions[i]).strip()
        pred_val = task['options'].get(pred, "N/A") if pred != "0" else "N/A"
        gt = gt_map.get(task['question'], "N/A")
        
        is_correct = False
        if gt != "N/A" and pred_val != "N/A":
            valid += 1
            if pred_val.replace(" ", "") == gt.replace(" ", ""):
                correct += 1
                is_correct = True
        
        rows.append({
            "StudentID": student_id,
            "Question": task['question'],
            "Category": task['category'],
            "Prediction": pred,
            "Pred_Val": pred_val,
            "Ground_Truth": gt,
            "Is_Correct": is_correct,
            "Method": method,
            "Exclude_Mode": "S3" if exclude_partial else ("S2" if exclude_target else "S1"),
            "Reasoning": reasons[i] if i < len(reasons) else ""
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(result_file, index=False, encoding='utf-8-sig')
    
    accuracy = (correct / valid * 100) if valid > 0 else 0
    return {'accuracy': accuracy, 'correct': correct, 'total': valid, 'cached': False}


def process_student_all_methods(student_id: str, methods: List[str], 
                                 targets: List[Dict], rebuild: bool,
                                 exclude_target: bool = False,
                                 exclude_partial: bool = False,
                                 result_suffix: str = "",
                                 output_dir: str = None,
                                 debug: bool = False) -> Dict:
    
    gt_map = load_ground_truth(Config.DATA_DIR, student_id, Config.TARGET_YEAR)
    
    student_results = {'student_id': student_id}
    for method in methods:
        result = run_single_method(
            student_id, method, targets, gt_map, rebuild,
            exclude_target=exclude_target,
            exclude_partial=exclude_partial,
            result_suffix=result_suffix,
            output_dir=output_dir,
            debug=debug
        )
        student_results[method] = result
    
    return student_results


def run_comparison(student_ids: List[str], methods: List[str], 
                   rebuild: bool = False, max_workers: int = 1,
                   exclude_target: bool = False, exclude_partial: bool = False,
                   output_dir: str = None, debug: bool = False):
    """
    Run comparison experiment across multiple students and methods.
    
    Settings:
    - S1 (Full-history): exclude_target=False, exclude_partial=False
    - S2 (Violence-blinded): exclude_target=True
    - S3 (S2 + Aggression): exclude_partial=True
    """
    
    base_output_dir = output_dir if output_dir else Config.RESULTS_DIR
    
    # Determine setting
    if exclude_partial:
        setting = "S3"  # S2 + Aggression
    elif exclude_target:
        setting = "S2"  # Violence-blinded
    else:
        setting = "S1"  # Full-history
    
    result_suffix = f"_{setting}"
    
    print("=" * 70)
    print("MIRROR Comparison Experiment")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Students: {len(student_ids)}")
    print(f"Methods: {methods}")
    print(f"Setting: {setting}")
    print(f"  - S1 (Full-history): All 2018-2022 responses")
    print(f"  - S2 (Violence-blinded): Exclude Aggression/Delinquency/Violence")
    print(f"  - S3 (S2 + Aggression): S2 with Aggression restored")
    print("=" * 70)
    
    # Load target questions
    all_targets = load_target_questions(Config.SCHEMA_FILE)
    targets = filter_target_questions(all_targets, partial_only=exclude_partial)
    
    if not targets:
        print("[Error] No target questions found")
        return {}
    
    print(f"Target questions: {len(targets)}")
    print()
    
    # Initialize results
    results = {method: {'correct': 0, 'total': 0, 'accuracies': []} for method in methods}
    
    # Process students
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_student_all_methods, sid, methods, targets, rebuild,
                    exclude_target, exclude_partial, result_suffix, base_output_dir, debug
                ): sid for sid in student_ids
            }
            
            with tqdm(total=len(student_ids), desc="Processing students") as pbar:
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        student_results = future.result()
                        for method in methods:
                            r = student_results.get(method, {})
                            results[method]['correct'] += r.get('correct', 0)
                            results[method]['total'] += r.get('total', 0)
                            results[method]['accuracies'].append(r.get('accuracy', 0))
                    except Exception as e:
                        print(f"\n[Error] {sid}: {e}")
                    pbar.update(1)
    else:
        for idx, sid in enumerate(student_ids):
            print(f"\n[{idx+1}/{len(student_ids)}] Student: {sid}")
            student_results = process_student_all_methods(
                sid, methods, targets, rebuild,
                exclude_target, exclude_partial, result_suffix, base_output_dir, debug
            )
            for method in methods:
                r = student_results.get(method, {})
                results[method]['correct'] += r.get('correct', 0)
                results[method]['total'] += r.get('total', 0)
                results[method]['accuracies'].append(r.get('accuracy', 0))
                cached_str = "(cached)" if r.get('cached') else ""
                print(f"  {method:15s}: {r.get('accuracy', 0):5.2f}% {cached_str}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY - Setting {setting}")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Δ':>8} {'Correct':>8} {'Total':>8}")
    print("-" * 70)
    
    summary_rows = []
    baseline_acc = None
    
    for method in methods:
        r = results[method]
        overall = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0
        avg = sum(r['accuracies']) / len(r['accuracies']) if r['accuracies'] else 0
        
        # Calculate delta from LLM_only baseline
        if method == 'LLM_only':
            baseline_acc = overall
            delta_str = "–"
        elif baseline_acc is not None:
            delta = overall - baseline_acc
            delta_str = f"{'↑' if delta > 0 else '↓'}{abs(delta):.2f}"
        else:
            delta_str = "–"
        
        print(f"{method:<20} {overall:>9.2f}% {delta_str:>8} {r['correct']:>8} {r['total']:>8}")
        
        summary_rows.append({
            'Setting': setting,
            'Method': method,
            'Accuracy': round(overall, 2),
            'Avg_Accuracy': round(avg, 2),
            'Delta': delta_str,
            'Correct': r['correct'],
            'Total': r['total'],
            'Students': len(r['accuracies'])
        })
    
    print("=" * 70)

    # Domain-wise macro accuracy (Paper Section 3.3, Table 6)
    print(f"\n{'=' * 70}")
    print(f"DOMAIN-WISE ACCURACY (Macro) - Setting {setting}")
    print("=" * 70)
    for method in methods:
        method_dir = os.path.join(base_output_dir, f"{method}_{setting}")
        if os.path.isdir(method_dir):
            domain_correct = {}
            domain_total = {}
            for f in sorted(os.listdir(method_dir)):
                if f.endswith(".csv") and f.startswith("pred_"):
                    try:
                        df_tmp = pd.read_csv(os.path.join(method_dir, f), encoding='utf-8-sig')
                        for _, row in df_tmp.iterrows():
                            cat = row.get('Category', 'Unknown')
                            if cat not in domain_correct:
                                domain_correct[cat] = 0
                                domain_total[cat] = 0
                            domain_total[cat] += 1
                            if row.get('Is_Correct', False):
                                domain_correct[cat] += 1
                    except Exception:
                        continue
            if domain_total:
                domain_accs = []
                print(f"\n  [{method}]")
                for cat in sorted(domain_total.keys()):
                    d_acc = 100 * domain_correct[cat] / domain_total[cat]
                    domain_accs.append(d_acc)
                    print(f"    {cat:<40} {d_acc:>6.2f}%")
                macro = sum(domain_accs) / len(domain_accs)
                print(f"    {'Macro Average':<40} {macro:>6.2f}%")
    print("=" * 70)

    # Save summary
    os.makedirs(base_output_dir, exist_ok=True)
    summary_file = os.path.join(base_output_dir, f"summary_{setting}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\nSummary saved: {summary_file}")

    return results


def run_ablation_study(student_ids: List[str], rebuild: bool = False,
                       max_workers: int = 1, output_dir: str = None):
    """
    Tests all method combinations across all settings.
    """
    
    ablation_methods = ['LLM_only', 'RER', 'RER_LTE', 'RER_KG', 'MIRROR']
    settings = [
        {'name': 'S1', 'exclude_target': False, 'exclude_partial': False},
        {'name': 'S2', 'exclude_target': True, 'exclude_partial': False},
        {'name': 'S3', 'exclude_target': False, 'exclude_partial': True},
    ]
    
    all_results = []
    
    for setting in settings:
        print(f"\n{'='*70}")
        print(f"Running Setting {setting['name']}")
        print(f"{'='*70}")
        
        results = run_comparison(
            student_ids=student_ids,
            methods=ablation_methods,
            rebuild=rebuild,
            max_workers=max_workers,
            exclude_target=setting['exclude_target'],
            exclude_partial=setting['exclude_partial'],
            output_dir=output_dir
        )
        
        for method, r in results.items():
            overall = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0
            all_results.append({
                'Setting': setting['name'],
                'Method': method,
                'Accuracy': round(overall, 2),
                'Correct': r['correct'],
                'Total': r['total']
            })
    
    # Save combined ablation results
    if output_dir:
        ablation_file = os.path.join(output_dir, "ablation_results.csv")
        pd.DataFrame(all_results).to_csv(ablation_file, index=False, encoding='utf-8-sig')
        print(f"\nAblation results saved: {ablation_file}")
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    
    df = pd.DataFrame(all_results)
    pivot = df.pivot(index='Method', columns='Setting', values='Accuracy')
    pivot = pivot.reindex(['LLM_only', 'RER', 'RER_LTE', 'RER_KG', 'MIRROR'])
    print(pivot.to_string())
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MIRROR Experiment Runner (Paper Implementation)")
    parser.add_argument("--student", type=str, help="Single student ID")
    parser.add_argument("--all", action="store_true", help="Run for all students")
    parser.add_argument("--methods", type=str, nargs='+', default=["MIRROR"],
                        choices=SUPPORTED_METHODS,
                        help="Methods to evaluate")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild cached results")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--exclude-target", action="store_true", help="Setting S2: Violence-blinded")
    parser.add_argument("--exclude-partial", action="store_true", help="Setting S3: S2 + Aggression")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Determine student IDs
    if args.all:
        student_ids = sorted(get_all_student_ids(Config.DATA_DIR))
    elif args.student:
        student_ids = [args.student]
    else:
        student_ids = sorted(get_all_student_ids(Config.DATA_DIR))[:1]
    
    print(f"Students to process: {len(student_ids)}")
    
    if args.ablation:
        # Run full ablation study
        run_ablation_study(
            student_ids=student_ids,
            rebuild=args.rebuild,
            max_workers=args.workers,
            output_dir=args.output_dir
        )
    else:
        # Run comparison with specified methods
        run_comparison(
            student_ids=student_ids,
            methods=args.methods,
            rebuild=args.rebuild,
            max_workers=args.workers,
            exclude_target=args.exclude_target,
            exclude_partial=args.exclude_partial,
            output_dir=args.output_dir,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
