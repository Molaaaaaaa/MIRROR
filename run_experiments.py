"""
Comparative Experiment Script
Filename: run_experiments.py

Supported Methods:
1. 2018_only - Use 2018 data only
2. 2022_only - Use 2022 data only
3. rag_only - RAG based prediction
4. rag_stm_ltm - RAG + STM + LTM (No KG)
5. MIRROR - RAG + STM + LTM + KG (Full Framework)
"""
import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_ollama import ChatOllama

from config import Config
from agent import create_agent
from utils import (
    load_student_data,
    load_ground_truth,
    load_target_questions,
    filter_target_questions,
    get_all_student_ids,
    clean_llm_output,
)

SUPPORTED_METHODS = ['2018_only', '2022_only', 'LLM_only', 'RER', 'RER_LTE', 'RER_KG', 'MIRROR']

METHOD_MAPPING = {
    'LLM_only': None,               # 도구 없음
    'RER': 'rag_only',              # RAG만
    'RER_LTE': 'rag_stm_ltm',       # RAG + LTM
    'RER_KG': 'rag_kg',             # RAG + KG
    'MIRROR': 'MIRROR',             # 전체
}

class BaselinePredictor:
    """Baseline Predictor for single-year methods"""
    
    def __init__(self, student_id: str, method: str, 
                 exclude_target: bool = False, exclude_partial: bool = False):
        self.student_id = student_id
        self.method = method
        
        _, self.history = load_student_data(
            Config.DATA_DIR, student_id, 
            exclude_target=exclude_target, 
            exclude_partial=exclude_partial
        )
        
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=30,
            num_ctx=Config.NUM_CTX,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
    
    def predict(self, question: str, options: Dict[str, str]) -> Tuple[str, str]:
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        if self.method == "2018_only":
            answer = self.history.get("2018", {}).get(question, "없음")
            context = f"[2018년] {answer}"
        elif self.method == "2022_only":
            answer = self.history.get("2022", {}).get(question, "없음")
            context = f"[2022년] {answer}"
        else:
            context = "Unknown"
        
        prompt = f"""{context}

질문: {question}
선택지:
{options_str}

2023년 답변 번호(1~6)만:"""
        
        try:
            response = self.llm.invoke(prompt)
            return clean_llm_output(response.content), self.method
        except:
            return "0", "Error"
    
    def predict_batch(self, tasks: List[Dict]) -> Tuple[List[str], List[str]]:
        preds, reasons = [], []
        for task in tasks:
            pred, reason = self.predict(task['question'], task['options'])
            preds.append(pred)
            reasons.append(reason)
        return preds, reasons


def run_single_method(student_id: str, method: str, targets: List[Dict], 
                      gt_map: Dict, rebuild: bool = False,
                      exclude_target: bool = False, exclude_partial: bool = False,
                      result_suffix: str = "", output_dir: str = None) -> Dict:
    """Run prediction for a single student with a specific method"""
    if method in METHOD_MAPPING:
        method = METHOD_MAPPING[method]

    base_dir = output_dir if output_dir else Config.RESULTS_DIR
    dir_suffix = result_suffix if result_suffix else ""
    result_dir = os.path.join(base_dir, f"{method}{dir_suffix}")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"pred_{student_id}.csv")
    
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
    
    # Method Selection
    if method in ["2018_only", "2022_only"]:
        predictor = BaselinePredictor(
            student_id, method, 
            exclude_target=exclude_target, 
            exclude_partial=exclude_partial
        )
        predictions, reasons = predictor.predict_batch(targets)
    
    elif method == "rag_only":
        agent = create_agent(
            student_id, "simplified", 
            exclude_target=exclude_target,
            exclude_partial=exclude_partial,
            tool_set="rag_only"
        )
        predictions, reasons = agent.predict_batch(targets)
    
    elif method == "rag_stm_ltm":
        agent = create_agent(
            student_id, "simplified",
            exclude_target=exclude_target,
            exclude_partial=exclude_partial,
            tool_set="rag_stm_ltm"
        )
        predictions, reasons = agent.predict_batch(targets)
    
    elif method == "MIRROR":
        agent = create_agent(
            student_id, "simplified",
            exclude_target=exclude_target,
            exclude_partial=exclude_partial,
            tool_set="full"
        )
        predictions, reasons = agent.predict_batch(targets)
    
    else:
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
            "Exclude_Mode": "partial" if exclude_partial else ("target" if exclude_target else "none"),
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
                                 output_dir: str = None) -> Dict:
    
    gt_map = load_ground_truth(Config.DATA_DIR, student_id, Config.TARGET_YEAR)
    
    student_results = {'student_id': student_id}
    for method in methods:
        result = run_single_method(
            student_id, method, targets, gt_map, rebuild,
            exclude_target=exclude_target,
            exclude_partial=exclude_partial,
            result_suffix=result_suffix,
            output_dir=output_dir
        )
        student_results[method] = result
    
    return student_results


def run_comparison(student_ids: List[str], methods: List[str], 
                   rebuild: bool = False, max_workers: int = 1,
                   exclude_target: bool = False, exclude_partial: bool = False,
                   output_dir: str = None):
    
    base_output_dir = output_dir if output_dir else Config.RESULTS_DIR
    
    if exclude_partial:
        exclude_mode = "exclude_partial"
    elif exclude_target:
        exclude_mode = "exclude_target"
    else:
        exclude_mode = "none"
    
    result_suffix = f"_{exclude_mode}" if exclude_mode != "none" else ""
    
    print("=" * 70)
    print("Comparison Experiment (Optimized)")
    print(f"Students: {len(student_ids)}")
    print(f"Methods: {methods}")
    print(f"Exclude Mode: {exclude_mode}")
    print("=" * 70)
    
    all_targets = load_target_questions(Config.SCHEMA_FILE)
    targets = filter_target_questions(all_targets, partial_only=exclude_partial)
    
    if not targets:
        print("[Error] No target questions")
        return {}
    
    results = {method: {'correct': 0, 'total': 0, 'accuracies': []} for method in methods}
    
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_student_all_methods, sid, methods, targets, rebuild,
                    exclude_target, exclude_partial, result_suffix, base_output_dir
                ): sid for sid in student_ids
            }
            
            with tqdm(total=len(student_ids), desc="Processing") as pbar:
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
            print(f"\n[{idx+1}/{len(student_ids)}] {sid}")
            student_results = process_student_all_methods(
                sid, methods, targets, rebuild,
                exclude_target, exclude_partial, result_suffix, base_output_dir
            )
            for method in methods:
                r = student_results.get(method, {})
                results[method]['correct'] += r.get('correct', 0)
                results[method]['total'] += r.get('total', 0)
                results[method]['accuracies'].append(r.get('accuracy', 0))
                print(f"  {method:15s}: {r.get('accuracy', 0):.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    
    summary_rows = []
    for method in methods:
        r = results[method]
        overall = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0
        avg = sum(r['accuracies']) / len(r['accuracies']) if r['accuracies'] else 0
        print(f"{method:20s}: {overall:.2f}% (avg: {avg:.2f}%)")
        
        summary_rows.append({
            'Method': method,
            'Overall_Accuracy': round(overall, 2),
            'Avg_Accuracy': round(avg, 2),
            'Correct': r['correct'],
            'Total': r['total']
        })
    
    os.makedirs(base_output_dir, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(base_output_dir, f"comparison_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--methods", type=str, nargs='+', default=["MIRROR"]) # Default changed
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exclude-target", action="store_true")
    parser.add_argument("--exclude-partial", action="store_true")
    
    args = parser.parse_args()
    
    if args.all:
        student_ids = sorted(get_all_student_ids(Config.DATA_DIR))
    elif args.student:
        student_ids = [args.student]
    else:
        student_ids = sorted(get_all_student_ids(Config.DATA_DIR))[:1]
    
    run_comparison(
        student_ids, 
        args.methods, 
        args.rebuild, 
        args.workers,
        exclude_target=args.exclude_target,
        exclude_partial=args.exclude_partial
    )

if __name__ == "__main__":
    main()