import os
import json
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from data_constants import SAMPLE_CATEGORIES_SIMILARITY

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class CategorySimilarityBuilder:
    
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
        print("Collecting questions from all students...")
        
        student_ids = get_all_student_ids(Config.DATA_DIR)
        all_questions: Dict[str, Set[str]] = defaultdict(set)
        
        for sid in tqdm(student_ids[:10], desc="Sampling students"):
            _, history = load_student_data(Config.DATA_DIR, sid, exclude_target=False)
            for year, year_data in history.items():
                for question in year_data.keys():
                    category = extract_category(question)
                    q_body = question.split(']')[-1].strip() if ']' in question else question
                    all_questions[category].add(q_body)
            
            from utils import load_ground_truth
            gt = load_ground_truth(Config.DATA_DIR, sid, Config.TARGET_YEAR)
            for question in gt.keys():
                category = extract_category(question)
                q_body = question.split(']')[-1].strip() if ']' in question else question
                all_questions[category].add(q_body)
        
        for category, questions in all_questions.items():
            self.category_questions[category] = list(questions)
        
        print(f"Collected {len(self.category_questions)} categories")
        for cat, qs in sorted(self.category_questions.items()):
            print(f"  - {cat}: {len(qs)} questions")
    
    def compute_category_embeddings(self):
        print("\nComputing category embeddings...")
        
        for category, questions in tqdm(self.category_questions.items(), desc="Embedding"):
            if not questions:
                continue
            
            texts = [f"{category}: {q}" for q in questions]
            
            embeddings = self.embeddings.embed_documents(texts)
            
            avg_embedding = np.mean(embeddings, axis=0)
            self.category_embeddings[category] = avg_embedding
        
        print(f"Computed embeddings for {len(self.category_embeddings)} categories")
    
    def compute_category_relationships(self, threshold: float = 0.3):
        print("\nComputing category relationships...")
        
        categories = list(self.category_embeddings.keys())
        n = len(categories)
        
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
                
                if sim >= threshold:
                    similarities[cat1].append((cat2, round(sim, 3)))
        
        for category in similarities:
            similarities[category].sort(key=lambda x: x[1], reverse=True)
            similarities[category] = similarities[category][:10]
        
        self.category_relationships = dict(similarities)
        
        total_relations = sum(len(v) for v in self.category_relationships.values())
        print(f"Found {total_relations} relationships (threshold >= {threshold})")
    
    def build(self) -> Dict:
        self.collect_all_questions()
        self.compute_category_embeddings()
        self.compute_category_relationships(threshold=0.3)
        
        return {
            "type": "category_embedding_similarity",
            "description": "Semantic similarity-based relationships between categories (question text embedding)",
            "category_count": len(self.category_questions),
            "category_relationships": self.category_relationships,
            "category_question_counts": {
                cat: len(qs) for cat, qs in self.category_questions.items()
            }
        }
    
    def save(self, output_path: str = None) -> str:
        if output_path is None:
            output_path = os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        kg_data = self.build()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved to: {output_path}")
        return output_path


def load_category_similarity(path: str = None) -> Dict:
    if path is None:
        path = os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
    
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
    
    output_path = args.output or os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
    
    data = {
        "type": "category_embedding_similarity",
        "description": "Semantic similarity-based relationships between categories (question text embedding)",
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
    
    print("\n" + "=" * 60)
    print("Sample Relationships:")
    print("=" * 60)
    
    sample_categories = SAMPLE_CATEGORIES_SIMILARITY
    for cat in sample_categories:
        if cat in builder.category_relationships:
            relations = builder.category_relationships[cat][:5]
            print(f"\n{cat}:")
            for related, sim in relations:
                print(f"  -> {related}: {sim}")
