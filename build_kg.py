import os
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from data_constants import (
    RESPONSE_SCALE,
    FREQUENCY_KEYWORDS,
    AGREEMENT_KEYWORDS,
)

from langchain_ollama import ChatOllama

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category


def get_numeric_value(response: str) -> Optional[float]:
    if response in RESPONSE_SCALE:
        return RESPONSE_SCALE[response]
    try:
        return float(response)
    except:
        return None


def calculate_trend(values: List[float]) -> str:
    if len(values) < 2:
        return 'insufficient'
    
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0
    
    if abs(slope) < 0.1:
        return 'stable'
    elif slope > 0:
        return 'increasing'
    else:
        return 'decreasing'


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


class KGBuilder:
    def __init__(self, student_id: str, llm: ChatOllama = None):
        self.student_id = student_id
        
        _, self.history = load_student_data(
            Config.DATA_DIR, student_id, exclude_target=False
        )
        self.history = {k: v for k, v in self.history.items() if int(k) < Config.TARGET_YEAR}
        
        self.llm = llm or ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=500,
            num_ctx=Config.NUM_CTX,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
        
        self.kg_data = {
            'student_id': student_id,
            'temporal_trends': {},
            'category_profiles': {},
            'category_relationships': {},
            'schema': {},
            'prediction_hints': {},
            'mirror _insights': {},
            'metadata': {
                'build_time': None,
                'years_covered': sorted([int(y) for y in self.history.keys()])
            }
        }
        
        self._category_yearly_values = defaultdict(lambda: defaultdict(list))
    
    def build_temporal_trends(self):
        all_questions = set()
        for year_data in self.history.values():
            all_questions.update(year_data.keys())
        
        for question in all_questions:
            series = []
            values = []
            category = extract_category(question)
            
            for year in sorted(self.history.keys()):
                if question in self.history[year]:
                    val = self.history[year][question]
                    series.append((int(year), val))
                    
                    numeric = get_numeric_value(val)
                    if numeric is not None:
                        values.append(numeric)
                        self._category_yearly_values[category][int(year)].append(numeric)
            
            if not series:
                continue
            
            trend_data = {
                'series': series,
                'category': category,
                'last_value': series[-1][1] if series else None,
            }
            
            if len(values) == len(series):
                trend_data['data_type'] = 'numeric'
                trend_data['trend'] = calculate_trend(values)
                
                if len(values) >= 2:
                    recent_diff = values[-1] - values[-2]
                    predicted = values[-1] + recent_diff * 0.5
                    trend_data['predicted_next'] = round(predicted, 1)
            else:
                trend_data['data_type'] = 'categorical'
                counter = Counter([s[1] for s in series])
                mode, count = counter.most_common(1)[0]
                trend_data['mode'] = mode
                trend_data['mode_ratio'] = count / len(series)
            
            self.kg_data['temporal_trends'][question] = trend_data
    
    def build_category_relationships(self):
        category_yearly_means = {}
        
        for category, yearly_values in self._category_yearly_values.items():
            yearly_means = {}
            for year, values in yearly_values.items():
                if values:
                    yearly_means[year] = sum(values) / len(values)
            
            if len(yearly_means) >= 3:
                category_yearly_means[category] = yearly_means
        
        categories = list(category_yearly_means.keys())
        
        correlations = defaultdict(list)
        
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                years1 = set(category_yearly_means[cat1].keys())
                years2 = set(category_yearly_means[cat2].keys())
                common_years = sorted(years1 & years2)
                
                if len(common_years) < 3:
                    continue
                
                values1 = [category_yearly_means[cat1][y] for y in common_years]
                values2 = [category_yearly_means[cat2][y] for y in common_years]
                
                corr = calculate_pearson_correlation(values1, values2)
                
                if abs(corr) >= 0.3:
                    correlations[cat1].append((cat2, round(corr, 3)))
                    correlations[cat2].append((cat1, round(corr, 3)))
        
        for category in correlations:
            correlations[category].sort(key=lambda x: abs(x[1]), reverse=True)
            correlations[category] = correlations[category][:5]
        
        self.kg_data['category_relationships'] = dict(correlations)
    
    def build_category_profiles(self):
        category_questions = defaultdict(list)
        
        for question, trend in self.kg_data['temporal_trends'].items():
            category = trend['category']
            category_questions[category].append({
                'question': question,
                'trend': trend.get('trend', 'unknown'),
                'last_value': trend.get('last_value'),
                'data_type': trend.get('data_type', 'unknown')
            })
        
        for category, questions in category_questions.items():
            trends = [q['trend'] for q in questions if q['trend'] != 'unknown']
            
            if trends:
                trend_counts = Counter(trends)
                dominant = trend_counts.most_common(1)[0][0]
            else:
                dominant = 'unknown'
            
            related = self.kg_data['category_relationships'].get(category, [])
            
            self.kg_data['category_profiles'][category] = {
                'question_count': len(questions),
                'dominant_trend': dominant,
                'related_categories': [r[0] for r in related[:3]],  # 상위 3개
                'sample_questions': [q['question'][:50] for q in questions[:3]]
            }
    
    def build_prediction_hints(self):
        for question, trend in self.kg_data['temporal_trends'].items():
            hints = []
            
            series = trend.get('series', [])
            if len(series) >= 3:
                last_values = [s[1] for s in series[-3:]]
                if len(set(last_values)) == 1:
                    hints.append(f"최근 3년 동일: '{last_values[0]}'")
            
            if trend.get('data_type') == 'numeric':
                trend_dir = trend.get('trend', '')
                if trend_dir == 'increasing':
                    hints.append("상승 추세")
                elif trend_dir == 'decreasing':
                    hints.append("하락 추세")
                
                predicted = trend.get('predicted_next')
                if predicted:
                    hints.append(f"예측값: {predicted}")
            
            elif trend.get('data_type') == 'categorical':
                mode = trend.get('mode', '')
                ratio = trend.get('mode_ratio', 0)
                if ratio >= 0.6:
                    hints.append(f"주된 응답({ratio:.0%}): '{mode}'")
            
            if hints:
                self.kg_data['prediction_hints'][question] = hints
    
    def build_mirror_insights(self):
        category_summaries = []
        for cat, profile in list(self.kg_data['category_profiles'].items())[:5]:
            trend = profile.get('dominant_trend', 'unknown')
            count = profile.get('question_count', 0)
            related = profile.get('related_categories', [])
            related_str = f", 관련: {','.join(related[:2])}" if related else ""
            category_summaries.append(f"- {cat}: {count}개 문항, {trend} 추세{related_str}")
        
        strong_correlations = []
        for cat, relations in self.kg_data['category_relationships'].items():
            for related_cat, corr in relations[:2]:
                if abs(corr) >= 0.5:
                    direction = "양의" if corr > 0 else "음의"
                    strong_correlations.append(f"- {cat} ↔ {related_cat}: {direction} 상관 ({corr})")
        
        sudden_changes = []
        for question, trend in self.kg_data['temporal_trends'].items():
            series = trend.get('series', [])
            if len(series) >= 3:
                recent = series[-1][1]
                prev_values = [s[1] for s in series[:-1]]
                if recent not in prev_values:
                    sudden_changes.append(f"- {question[:40]}...: {prev_values[-1]} -> {recent}")
        
        prompt = f"""학생 {self.student_id}의 설문 데이터를 분석하세요.

[카테고리별 현황]
{chr(10).join(category_summaries[:10])}

[카테고리간 상관관계]
{chr(10).join(strong_correlations[:5]) if strong_correlations else '특이 상관 없음'}

[2022년 급변 항목 ({len(sudden_changes)}개)]
{chr(10).join(sudden_changes[:5]) if sudden_changes else '없음'}

다음을 분석하세요:
1. 전반적 발달 특성
2. 카테고리간 관계 패턴
3. 2023년 예측 시 고려사항

3-4문장으로 요약:"""
        
        try:
            response = self.llm.invoke(prompt)
            self.kg_data['mirror_insights'] = {
                'raw_analysis': response.content.strip(),
                'sudden_change_count': len(sudden_changes),
                'strong_correlation_count': len([c for c in strong_correlations if True])
            }
        except Exception as e:
            self.kg_data['mirror_insights'] = {
                'raw_analysis': f"분석 실패: {str(e)[:50]}",
                'sudden_change_count': len(sudden_changes),
                'strong_correlation_count': 0
            }

    def build_schema_structure(self):
        if not os.path.exists(Config.SCHEMA_FILE):
            print(f"[Warning] Schema file not found: {Config.SCHEMA_FILE}")
            return
        
        with open(Config.SCHEMA_FILE, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        option_nodes = {}
        for topic, options in schema_data.items():
            if isinstance(options, list) and options:
                option_type = self._detect_option_type_from_list(options)
                option_nodes[topic] = {
                    'options': {str(i+1): opt for i, opt in enumerate(options)},
                    'option_count': len(options),
                    'option_type': option_type
                }
        
        topic_to_questions = defaultdict(list)
        for question, trend in self.kg_data.get('temporal_trends', {}).items():
            topic = trend.get('category', '')
            if topic:
                topic_to_questions[topic].append(question)
        
        self.kg_data['schema'] = {
            'option_nodes': option_nodes,
            'topic_to_questions': dict(topic_to_questions),
            'statistics': {
                'topic_count': len(option_nodes),
                'question_count': sum(len(qs) for qs in topic_to_questions.values()),
                'option_types': {
                    'frequency': sum(1 for v in option_nodes.values() if v['option_type'] == 'frequency'),
                    'agreement': sum(1 for v in option_nodes.values() if v['option_type'] == 'agreement'),
                    'other': sum(1 for v in option_nodes.values() if v['option_type'] == 'other')
                }
            }
        }
        
        print(f"[Schema] {len(option_nodes)} topics, "
            f"{self.kg_data['schema']['statistics']['question_count']} questions")


    def _detect_option_type_from_list(self, options: List[str]) -> str:
        text = ' '.join(options)
        freq_score = sum(1 for kw in FREQUENCY_KEYWORDS if kw in text)
        agree_score = sum(1 for kw in AGREEMENT_KEYWORDS if kw in text)
        
        if freq_score > agree_score:
            return 'frequency'
        elif agree_score > 0:
            return 'agreement'
        return 'other'
    
    def build_all(self) -> Dict[str, Any]:
        if not self.history:
            return self.kg_data
        
        self.build_temporal_trends()
        
        self.build_category_relationships()
        
        self.build_category_profiles()
        self.build_prediction_hints()
        
        self.build_schema_structure()
        
        self.build_mirror_insights()
        
        self.kg_data['metadata']['build_time'] = datetime.now().isoformat()
        return self.kg_data
    
    def save(self, output_dir: str = None) -> str:
        if output_dir is None:
            output_dir = Config.MIRROR_MEMORY_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.student_id}_kg.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.kg_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    


def build_kg_for_student(student_id: str, llm: ChatOllama = None,
                         output_dir: str = None) -> Tuple[str, float, Dict]:
    import time
    start = time.time()
    
    builder = KGBuilder(student_id, llm=llm)
    builder.build_all()
    output_path = builder.save(output_dir)
    
    elapsed = time.time() - start
    
    stats = {
        'trends': len(builder.kg_data['temporal_trends']),
        'categories': len(builder.kg_data['category_profiles']),
        'relationships': len(builder.kg_data['category_relationships']),
        'hints': len(builder.kg_data['prediction_hints'])
    }
    
    return output_path, elapsed, stats



def build_kg_batch(student_ids: List[str], output_dir: str = None,
                   max_workers: int = 2, skip_existing: bool = False):
    import time
    
    print(f"Building KG for {len(student_ids)} students...")
    print(f"Workers: {max_workers}, Skip existing: {skip_existing}")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = Config.MIRROR_MEMORY_DIR
    
    if skip_existing:
        existing = set()
        for f in os.listdir(output_dir) if os.path.exists(output_dir) else []:
            if f.endswith('_kg.json'):
                sid = f.replace('_kg.json', '')
                existing.add(sid)
        
        original = len(student_ids)
        student_ids = [sid for sid in student_ids if sid not in existing]
        print(f"Skipping {original - len(student_ids)} existing files")
    
    if not student_ids:
        print("No students to process")
        return []
    
    results = []
    total_time = 0
    
    def process_student(sid):
        llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=500,
            num_ctx=Config.NUM_CTX,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
        return sid, build_kg_for_student(sid, llm=llm, output_dir=output_dir)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_student, sid): sid for sid in student_ids}
        
        with tqdm(total=len(student_ids), desc="Building KG") as pbar:
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    _, (path, elapsed, stats) = future.result()
                    results.append((sid, True, elapsed, stats))
                    total_time += elapsed
                    pbar.set_postfix({
                        'last': f"{sid}({elapsed:.1f}s)",
                        'relations': stats['relationships']
                    })
                except Exception as e:
                    results.append((sid, False, 0, str(e)))
                pbar.update(1)
    
    success = sum(1 for r in results if r[1])
    print("=" * 60)
    print(f"Complete: {success} success, {len(results) - success} failed")
    print(f"Total time: {total_time/60:.1f} min")
    
    if results:
        success_results = [r for r in results if r[1]]
        if success_results:
            avg_relations = sum(r[3]['relationships'] for r in success_results) / len(success_results)
            print(f"Avg relationships per student: {avg_relations:.1f}")
    
    return results


def load_student_kg(student_id: str, kg_dir: str = None) -> Optional[Dict]:
    if kg_dir is None:
        kg_dir = Config.MIRROR_MEMORY_DIR
    
    kg_path = os.path.join(kg_dir, f"{student_id}_kg.json")
    if os.path.exists(kg_path):
        with open(kg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph")
    parser.add_argument("--student", type=str, help="Single student ID")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--student-range", type=int, nargs=2, metavar=('START', 'END'))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all (ignore existing)")
    args = parser.parse_args()
    
    skip = args.skip_existing and not args.rebuild
    
    if args.all or args.student_range:
        all_ids = sorted(get_all_student_ids(Config.DATA_DIR))
        
        if args.student_range:
            start, end = args.student_range
            student_ids = all_ids[max(0, start):min(len(all_ids), end + 1)]
            print(f"[Range] {start}-{end}: {len(student_ids)} students")
        else:
            student_ids = all_ids
        
        build_kg_batch(
            student_ids,
            output_dir=args.output_dir,
            max_workers=args.workers,
            skip_existing=skip
        )
    elif args.student:
        path, elapsed, stats = build_kg_for_student(args.student, output_dir=args.output_dir)
        print(f"[{args.student}] Built in {elapsed:.1f}s")
        print(f"  Stats: {stats}")
        print(f"  Saved to: {path}")
    else:
        print("Usage:")
        print("  python build_kg.py --student 10001")
        print("  python build_kg.py --all --workers 2")
        print("  python build_kg.py --all --rebuild")
