"""
행동 상관성 기반 카테고리 관계 빌드 스크립트
파일명: build_behavioral_correlation.py

전체 학생의 2018-2022년 응답 데이터를 기반으로
카테고리 간 행동 상관성(Behavioral Correlation)을 계산

핵심 차이점:
- 텍스트 임베딩 유사도: "친구가 많다" ↔ "외톨이다" = 유사 (같은 주제)
- 행동 상관성: "친구가 많다" ↔ "외톨이다" = 음의 상관 (반대 응답)

학술적 의의:
- Behavior-Aware Retrieval
- 텍스트 기반 RAG의 한계를 도메인 지식으로 극복
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category


RESPONSE_SCALE = {
    '전혀 그렇지 않다': 1, '그렇지 않은 편이다': 2, '그런 편이다': 3, '매우 그렇다': 4,
    '전혀 없다': 1, '1~2번': 2, '3~5번': 3, '6번 이상': 4,
    '전혀 하지 않는다': 1, '거의 하지 않는다': 2, '가끔 한다': 3, '자주 한다': 4,
    '전혀 안함': 0, '30분 미만': 1, '30분 ~ 1시간 미만': 2, '1시간 ~ 2시간 미만': 3,
    '2시간 ~ 3시간 미만': 4, '3시간 ~ 4시간 미만': 5, '4시간 이상~': 6,
    '아주 불행한 사람이다': 1, '불행한 사람이다': 2, '행복한 사람이다': 3, '아주 행복한 사람이다': 4,
    '매우 못 잔다': 1, '못 자는 편이다': 2, '잘 자는 편이다': 3, '매우 잘 잔다': 4,
}


def get_numeric_value(response: str) -> Optional[float]:
    """응답을 수치로 변환"""
    if response in RESPONSE_SCALE:
        return float(RESPONSE_SCALE[response])
    try:
        return float(response)
    except:
        return None


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """피어슨 상관계수 계산"""
    if len(x) != len(y) or len(x) < 10:  # 최소 10개 샘플 필요
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
    """전역 행동 상관성 빌더"""
    
    def __init__(self):
        self.student_ids = get_all_student_ids(Config.DATA_DIR)
        
        # 카테고리별 학생-응답 데이터 저장
        # {category: {student_id: [연도별 평균값들]}}
        self.category_student_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # 최종 결과
        self.category_relationships: Dict[str, List[Tuple[str, float]]] = {}
        
    def collect_all_responses(self):
        """전체 학생 데이터 수집"""
        print(f"Collecting responses from {len(self.student_ids)} students...")
        
        for sid in tqdm(self.student_ids, desc="Loading students"):
            _, history = load_student_data(Config.DATA_DIR, sid, exclude_target=False)
            
            # 2018-2022년 데이터만 사용
            history = {k: v for k, v in history.items() if int(k) < Config.TARGET_YEAR}
            
            if not history:
                continue
            
            # 카테고리별 연도별 평균값 계산
            category_yearly: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
            
            for year_str, year_data in history.items():
                year = int(year_str)
                for question, response in year_data.items():
                    category = extract_category(question)
                    numeric_val = get_numeric_value(response)
                    if numeric_val is not None:
                        category_yearly[category][year].append(numeric_val)
            
            # 카테고리별 연도 평균 계산
            for category, yearly_data in category_yearly.items():
                yearly_means = []
                for year in sorted(yearly_data.keys()):
                    values = yearly_data[year]
                    if values:
                        yearly_means.append(sum(values) / len(values))
                
                if yearly_means:
                    # 전체 연도 평균을 학생의 대표값으로 사용
                    student_mean = sum(yearly_means) / len(yearly_means)
                    self.category_student_values[category][sid].append(student_mean)
        
        print(f"Collected {len(self.category_student_values)} categories")
        for cat in sorted(self.category_student_values.keys())[:10]:
            n_students = len(self.category_student_values[cat])
            print(f"  - {cat}: {n_students} students")
    
    def compute_category_correlations(self, min_samples: int = 30, threshold: float = 0.2):
        """카테고리 간 행동 상관성 계산"""
        print(f"\nComputing behavioral correlations (min_samples={min_samples}, threshold={threshold})...")
        
        categories = list(self.category_student_values.keys())
        n = len(categories)
        
        # 각 카테고리의 학생별 대표값 (평균) 계산
        category_vectors: Dict[str, Dict[str, float]] = {}
        
        for cat in categories:
            student_values = self.category_student_values[cat]
            category_vectors[cat] = {
                sid: sum(vals) / len(vals) 
                for sid, vals in student_values.items() 
                if vals
            }
        
        # 모든 카테고리 쌍에 대해 상관계수 계산
        correlations: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        for i in tqdm(range(n), desc="Computing correlations"):
            cat1 = categories[i]
            vec1 = category_vectors[cat1]
            
            for j in range(n):
                if i == j:
                    continue
                
                cat2 = categories[j]
                vec2 = category_vectors[cat2]
                
                # 공통 학생 찾기
                common_students = set(vec1.keys()) & set(vec2.keys())
                
                if len(common_students) < min_samples:
                    continue
                
                # 상관계수 계산
                x = [vec1[sid] for sid in common_students]
                y = [vec2[sid] for sid in common_students]
                
                corr = calculate_pearson_correlation(x, y)
                
                if corr is not None and abs(corr) >= threshold:
                    correlations[cat1].append((cat2, round(corr, 3)))
        
        # 상관계수 절대값 기준 정렬
        for category in correlations:
            correlations[category].sort(key=lambda x: abs(x[1]), reverse=True)
            correlations[category] = correlations[category][:10]  # 상위 10개
        
        self.category_relationships = dict(correlations)
        
        # 통계 출력
        total_relations = sum(len(v) for v in self.category_relationships.values())
        print(f"Found {total_relations} relationships")
    
    def build(self) -> Dict:
        """전체 빌드 실행"""
        self.collect_all_responses()
        self.compute_category_correlations()
        
        return {
            "type": "behavioral_correlation",
            "description": "카테고리 간 행동 상관성 (전체 학생 응답 패턴 기반)",
            "category_count": len(self.category_relationships),
            "category_relationships": self.category_relationships,
            "student_count": len(self.student_ids)
        }
    
    def save(self, output_path: str = None) -> str:
        """결과 저장"""
        if output_path is None:
            output_path = os.path.join(Config.AGENT_MEMORY_DIR, "behavioral_correlation.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = self.build()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved to: {output_path}")
        return output_path


def load_behavioral_correlation(path: str = None) -> Dict:
    """행동 상관성 데이터 로드"""
    if path is None:
        path = os.path.join(Config.AGENT_MEMORY_DIR, "behavioral_correlation.json")
    
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
    
    output_path = args.output or os.path.join(Config.AGENT_MEMORY_DIR, "behavioral_correlation.json")
    
    data = {
        "type": "behavioral_correlation",
        "description": "카테고리 간 행동 상관성 (전체 학생 응답 패턴 기반)",
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
    
    # 결과 미리보기
    print("\n" + "=" * 60)
    print("Sample Behavioral Correlations:")
    print("=" * 60)
    
    sample_categories = ["공격성", "학교 폭력", "사회적 위축", "우울", "자아존중감"]
    for cat in sample_categories:
        if cat in builder.category_relationships:
            relations = builder.category_relationships[cat][:5]
            print(f"\n{cat}:")
            for related, corr in relations:
                direction = "+" if corr > 0 else ""
                print(f"  -> {related}: {direction}{corr}")
        else:
            print(f"\n{cat}: (데이터 없음)")
