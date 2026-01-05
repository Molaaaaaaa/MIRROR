"""
카테고리 임베딩 유사도 빌드 스크립트
파일명: build_category_similarity.py

모든 카테고리(2018-2023)의 질문 텍스트를 임베딩하여
카테고리 간 의미적 유사도 기반 관계를 계산

- 개인 KG: 개인 응답 데이터 기반 상관관계 (2018-2022만 존재하는 카테고리)
- 카테고리 유사도: 질문 텍스트 기반 임베딩 유사도 (2023 신규 카테고리 포함)
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class CategorySimilarityBuilder:
    """카테고리 임베딩 유사도 빌더"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="SamilPwC-AXNode-GenAI/PwC-Embedding_expr",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.category_questions: Dict[str, List[str]] = defaultdict(list)
        self.category_embeddings: Dict[str, np.ndarray] = {}
        self.category_relationships: Dict[str, List[Tuple[str, float]]] = {}
        
    def collect_all_questions(self):
        """모든 학생 데이터에서 질문 수집 (2018-2023)"""
        print("Collecting questions from all students...")
        
        student_ids = get_all_student_ids(Config.DATA_DIR)
        all_questions: Dict[str, Set[str]] = defaultdict(set)
        
        for sid in tqdm(student_ids[:10], desc="Sampling students"):
            # 2018-2022 데이터
            _, history = load_student_data(Config.DATA_DIR, sid, exclude_target=False)
            for year, year_data in history.items():
                for question in year_data.keys():
                    category = extract_category(question)
                    # 질문에서 카테고리 부분 제거하고 본문만 저장
                    q_body = question.split(']')[-1].strip() if ']' in question else question
                    all_questions[category].add(q_body)
            
            # 2023 데이터도 로드 (Ground Truth에서)
            from utils import load_ground_truth
            gt = load_ground_truth(Config.DATA_DIR, sid, Config.TARGET_YEAR)
            for question in gt.keys():
                category = extract_category(question)
                q_body = question.split(']')[-1].strip() if ']' in question else question
                all_questions[category].add(q_body)
        
        # Set을 List로 변환
        for category, questions in all_questions.items():
            self.category_questions[category] = list(questions)
        
        print(f"Collected {len(self.category_questions)} categories")
        for cat, qs in sorted(self.category_questions.items()):
            print(f"  - {cat}: {len(qs)} questions")
    
    def compute_category_embeddings(self):
        """카테고리별 평균 임베딩 계산"""
        print("\nComputing category embeddings...")
        
        for category, questions in tqdm(self.category_questions.items(), desc="Embedding"):
            if not questions:
                continue
            
            # 카테고리명 + 질문 텍스트 결합
            texts = [f"{category}: {q}" for q in questions]
            
            # 임베딩 계산
            embeddings = self.embeddings.embed_documents(texts)
            
            # 평균 임베딩
            avg_embedding = np.mean(embeddings, axis=0)
            self.category_embeddings[category] = avg_embedding
        
        print(f"Computed embeddings for {len(self.category_embeddings)} categories")
    
    def compute_category_relationships(self, threshold: float = 0.3):
        """카테고리 간 유사도 계산"""
        print("\nComputing category relationships...")
        
        categories = list(self.category_embeddings.keys())
        n = len(categories)
        
        # 모든 카테고리 쌍에 대해 유사도 계산
        similarities = defaultdict(list)
        
        for i in tqdm(range(n), desc="Computing similarities"):
            cat1 = categories[i]
            emb1 = self.category_embeddings[cat1]
            
            for j in range(n):
                if i == j:
                    continue
                    
                cat2 = categories[j]
                emb2 = self.category_embeddings[cat2]
                
                sim = cosine_similarity(emb1, emb2)
                
                # threshold 이상인 경우만 저장
                if sim >= threshold:
                    similarities[cat1].append((cat2, round(sim, 3)))
        
        # 유사도 기준 정렬 및 상위 N개 유지
        for category in similarities:
            similarities[category].sort(key=lambda x: x[1], reverse=True)
            similarities[category] = similarities[category][:10]  # 상위 10개
        
        self.category_relationships = dict(similarities)
        
        # 통계 출력
        total_relations = sum(len(v) for v in self.category_relationships.values())
        print(f"Found {total_relations} relationships (threshold >= {threshold})")
    
    def build(self) -> Dict:
        """전역 KG 빌드"""
        self.collect_all_questions()
        self.compute_category_embeddings()
        self.compute_category_relationships(threshold=0.3)
        
        return {
            "type": "category_embedding_similarity",
            "description": "카테고리 간 의미적 유사도 기반 관계 (질문 텍스트 임베딩)",
            "category_count": len(self.category_questions),
            "category_relationships": self.category_relationships,
            "category_question_counts": {
                cat: len(qs) for cat, qs in self.category_questions.items()
            }
        }
    
    def save(self, output_path: str = None) -> str:
        """카테고리 유사도 저장"""
        if output_path is None:
            output_path = os.path.join(Config.AGENT_MEMORY_DIR, "category_similarity.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        kg_data = self.build()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved to: {output_path}")
        return output_path


def load_category_similarity(path: str = None) -> Dict:
    """카테고리 유사도 로드"""
    if path is None:
        path = os.path.join(Config.AGENT_MEMORY_DIR, "category_similarity.json")
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Category Embedding Similarity")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Similarity threshold")
    args = parser.parse_args()
    
    builder = CategorySimilarityBuilder()
    builder.collect_all_questions()
    builder.compute_category_embeddings()
    builder.compute_category_relationships(threshold=args.threshold)
    
    output_path = args.output or os.path.join(Config.AGENT_MEMORY_DIR, "category_similarity.json")
    
    data = {
        "type": "category_embedding_similarity",
        "description": "카테고리 간 의미적 유사도 기반 관계 (질문 텍스트 임베딩)",
        "threshold": args.threshold,
        "category_count": len(builder.category_questions),
        "category_relationships": builder.category_relationships,
        "category_question_counts": {
            cat: len(qs) for cat, qs in builder.category_questions.items()
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to: {output_path}")
    
    # 결과 미리보기
    print("\n" + "=" * 60)
    print("Sample Relationships:")
    print("=" * 60)
    
    sample_categories = ["학교 폭력", "공격성", "사회적 위축", "우울"]
    for cat in sample_categories:
        if cat in builder.category_relationships:
            relations = builder.category_relationships[cat][:5]
            print(f"\n{cat}:")
            for related, sim in relations:
                print(f"  -> {related}: {sim}")
