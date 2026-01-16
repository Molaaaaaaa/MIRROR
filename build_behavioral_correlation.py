import os
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category
from data_constants import (
    RESPONSE_SCALE,
    SAMPLE_CATEGORIES_BEHAVIOR,
)

def get_numeric_value(response: str) -> Optional[float]:
    if response in RESPONSE_SCALE:
        return float(RESPONSE_SCALE[response])
    try:
        return float(response)
    except:
        return None


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 10:
        return None
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return None
    
    return numerator / denominator


class BehavioralCorrelationBuilder:
    def __init__(self):
        self.student_ids = get_all_student_ids(Config.DATA_DIR)
        
        self.category_student_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        self.category_relationships: Dict[str, List[Tuple[str, float]]] = {}
        
    def collect_all_responses(self):
        print(f"Collecting responses from {len(self.student_ids)} students...")
        
        for sid in tqdm(self.student_ids, desc="Loading students"):
            _, history = load_student_data(Config.DATA_DIR, sid, exclude_target=False)
            
            history = {k: v for k, v in history.items() if int(k) < Config.TARGET_YEAR}
            
            if not history:
                continue
            
            category_yearly: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
            
            for year_str, year_data in history.items():
                year = int(year_str)
                for question, response in year_data.items():
                    category = extract_category(question)
                    numeric_val = get_numeric_value(response)
                    if numeric_val is not None:
                        category_yearly[category][year].append(numeric_val)
            
            for category, yearly_data in category_yearly.items():
                yearly_means = []
                for year in sorted(yearly_data.keys()):
                    values = yearly_data[year]
                    if values:
                        yearly_means.append(sum(values) / len(values))
                
                if yearly_means:
                    student_mean = sum(yearly_means) / len(yearly_means)
                    self.category_student_values[category][sid].append(student_mean)
        
        print(f"Collected {len(self.category_student_values)} categories")
        for cat in sorted(self.category_student_values.keys())[:10]:
            n_students = len(self.category_student_values[cat])
            print(f"  - {cat}: {n_students} students")
    
    def compute_category_correlations(self, min_samples: int = 30, threshold: float = 0.2):
        print(f"\nComputing behavioral correlations (min_samples={min_samples}, threshold={threshold})...")
        
        categories = list(self.category_student_values.keys())
        n = len(categories)
        
        category_vectors: Dict[str, Dict[str, float]] = {}
        
        for cat in categories:
            student_values = self.category_student_values[cat]
            category_vectors[cat] = {
                sid: sum(vals) / len(vals) 
                for sid, vals in student_values.items() 
                if vals
            }
        
        correlations: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        for i in tqdm(range(n), desc="Computing correlations"):
            cat1 = categories[i]
            vec1 = category_vectors[cat1]
            
            for j in range(n):
                if i == j:
                    continue
                
                cat2 = categories[j]
                vec2 = category_vectors[cat2]
                
                common_students = set(vec1.keys()) & set(vec2.keys())
                
                if len(common_students) < min_samples:
                    continue
                
                x = [vec1[sid] for sid in common_students]
                y = [vec2[sid] for sid in common_students]
                
                corr = calculate_pearson_correlation(x, y)
                
                if corr is not None and abs(corr) >= threshold:
                    correlations[cat1].append((cat2, round(corr, 3)))
        
        for category in correlations:
            correlations[category].sort(key=lambda x: abs(x[1]), reverse=True)
            correlations[category] = correlations[category][:10]
        
        self.category_relationships = dict(correlations)
        
        total_relations = sum(len(v) for v in self.category_relationships.values())
        print(f"Found {total_relations} relationships")
    
    def build(self) -> Dict:
        self.collect_all_responses()
        self.compute_category_correlations()
        
        return {
            "type": "behavioral_correlation",
            "description": "Behavioral correlation between categories (based on all student response patterns)",
            "category_count": len(self.category_relationships),
            "category_relationships": self.category_relationships,
            "student_count": len(self.student_ids)
        }
    
    def save(self, output_path: str = None) -> str:
        if output_path is None:
            output_path = os.path.join(Config.MIRROR_MEMORY_DIR, "behavioral_correlation.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = self.build()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved to: {output_path}")
        return output_path


def load_behavioral_correlation(path: str = None) -> Dict:
    if path is None:
        path = os.path.join(Config.MIRROR_MEMORY_DIR, "behavioral_correlation.json")
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Behavioral Correlation")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum samples for correlation")
    parser.add_argument("--threshold", type=float, default=0.2, help="Correlation threshold")
    args = parser.parse_args()
    
    builder = BehavioralCorrelationBuilder()
    builder.collect_all_responses()
    builder.compute_category_correlations(
        min_samples=args.min_samples,
        threshold=args.threshold
    )
    
    output_path = args.output or os.path.join(Config.MIRROR_MEMORY_DIR, "behavioral_correlation.json")
    
    data = {
        "type": "behavioral_correlation",
        "description": "Behavioral correlation between categories (based on all student response patterns)",
        "min_samples": args.min_samples,
        "threshold": args.threshold,
        "category_count": len(builder.category_relationships),
        "category_relationships": builder.category_relationships,
        "student_count": len(builder.student_ids)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Sample Behavioral Correlations:")
    print("=" * 60)
    
    sample_categories = SAMPLE_CATEGORIES_BEHAVIOR
    for cat in sample_categories:
        if cat in builder.category_relationships:
            relations = builder.category_relationships[cat][:5]
            print(f"\n{cat}:")
            for related, corr in relations:
                direction = "+" if corr > 0 else ""
                print(f"  -> {related}: {direction}{corr}")
        else:
            print(f"\n{cat}: (No data available)")
