import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from data_constants import (
    RESPONSE_SCALE,
    NEGATIVE_KEYWORDS,
    INPUT_VARIABLE_NAMES,
    CATEGORY_HAPPINESS,
    CATEGORY_DEPRESSION,
    WELLBEING_CATEGORIES,
)

from langchain_ollama import ChatOllama

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category

def get_numeric_value(response: str) -> Optional[float]:
    if response in RESPONSE_SCALE:
        return RESPONSE_SCALE[response]
    try:
        return float(response)
    except:
        return None


def is_negative_direction(question: str) -> bool:
    return any(kw in question for kw in NEGATIVE_KEYWORDS)


def calculate_trend(values: List[float]) -> Dict[str, Any]:
    if len(values) < 2:
        return {'direction': 'insufficient', 'magnitude': 0, 'recent_change': 0}
    
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0
    recent_change = values[-1] - values[-2] if len(values) >= 2 else 0
    
    if abs(slope) < 0.1:
        direction = 'stable'
    elif slope > 0:
        direction = 'increasing'
    else:
        direction = 'decreasing'
    
    return {
        'direction': direction,
        'slope': round(slope, 3),
        'recent_change': round(recent_change, 2),
        'first_value': values[0],
        'last_value': values[-1],
        'range': round(max(values) - min(values), 2)
    }


def detect_sudden_shift(values: List[Any], years: List[int]) -> Optional[Dict]:
    if len(values) < 3:
        return None
    
    last_val = values[-1]
    prev_vals = values[:-1]
    
    counter = Counter(prev_vals)
    mode_val, mode_count = counter.most_common(1)[0]
    mode_ratio = mode_count / len(prev_vals)
    
    if mode_ratio >= 0.6 and last_val != mode_val:
        return {
            'detected': True,
            'year': years[-1],
            'from_value': mode_val,
            'to_value': last_val,
            'stability_before': round(mode_ratio, 2),
            'type': 'sudden_shift_from_stable'
        }
    
    numeric_vals = [get_numeric_value(str(v)) for v in values]
    if all(v is not None for v in numeric_vals):
        prev_numeric = numeric_vals[:-1]
        last_numeric = numeric_vals[-1]
        
        mean = sum(prev_numeric) / len(prev_numeric)
        variance = sum((x - mean) ** 2 for x in prev_numeric) / len(prev_numeric)
        std = variance ** 0.5
        
        if std > 0:
            z_score = abs(last_numeric - mean) / std
            if z_score > 2.0:
                return {
                    'detected': True,
                    'year': years[-1],
                    'from_value': mode_val,
                    'to_value': last_val,
                    'z_score': round(z_score, 2),
                    'type': 'statistical_outlier'
                }
    
    return None


class LTMBuilder:
    def __init__(self, student_id: str, llm: ChatOllama = None):
        self.student_id = student_id
        
        self.input_vars, self.history = load_student_data(
            Config.DATA_DIR, student_id, exclude_target=False
        )
        self.history = {k: v for k, v in self.history.items() if int(k) < Config.TARGET_YEAR}
        
        self.llm = llm or ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=800,
            num_ctx=Config.NUM_CTX,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
        
        self.ltm_data = {
            'student_id': student_id,
            'static_persona': {},
            'temporal_patterns': {},
            'thematic_profiles': {},
            'yearly_changes': {},
            'sudden_shifts': {},
            'consistency_issues': [],
            'prediction_hints': {},
            'category_analysis': {},
            'overall_narrative': '',
            'data_quality': {},
            'metadata': {
                'build_time': None,
                'years_covered': sorted([int(y) for y in self.history.keys()]),
                'method': 'ltm_v3'
            }
        }

    def build_static_persona(self):
        self.ltm_data['static_persona'] = {
            'gender': self.input_vars.get(INPUT_VARIABLE_NAMES['gender'], ''),
            'birth_year': self.input_vars.get(INPUT_VARIABLE_NAMES['birth_year'], ''),
            'region': self.input_vars.get(INPUT_VARIABLE_NAMES.get('residence_region', ''), ''),
            'school_region': self.input_vars.get(INPUT_VARIABLE_NAMES['school_region'], ''),
            'city_size': self.input_vars.get(INPUT_VARIABLE_NAMES['city_size'], ''),
            'siblings': self.input_vars.get(INPUT_VARIABLE_NAMES['siblings'], ''),
            'raw': self.input_vars
        }
        
        valid_fields = sum(1 for v in self.ltm_data['static_persona'].values() 
                        if v and v != {} and v != '')
        print(f"[Static Persona] {valid_fields} demographic fields loaded")
    
    def build_temporal_patterns(self):
        all_questions = set()
        for year_data in self.history.values():
            all_questions.update(year_data.keys())
        
        sudden_shift_count = 0
        
        for question in all_questions:
            series = []
            topic = extract_category(question)
            years = []
            values = []
            
            for year in sorted(self.history.keys()):
                if question in self.history[year]:
                    val = self.history[year][question]
                    series.append({'year': int(year), 'value': val})
                    years.append(int(year))
                    values.append(val)
            
            if not series:
                continue
            
            unique_values = set(values)
            counter = Counter(values)
            most_common_val, most_common_count = counter.most_common(1)[0]
            mode_ratio = most_common_count / len(values)
            
            if len(unique_values) == 1:
                stability = 'constant'
            elif mode_ratio >= 0.8:
                stability = 'highly_stable'
            elif mode_ratio >= 0.6:
                stability = 'stable'
            else:
                stability = 'variable'
            
            consecutive_same = 1
            for i in range(len(values) - 2, -1, -1):
                if values[i] == values[-1]:
                    consecutive_same += 1
                else:
                    break
            
            sudden_shift = detect_sudden_shift(values, years)
            if sudden_shift and sudden_shift['detected']:
                self.ltm_data['sudden_shifts'][question] = sudden_shift
                sudden_shift_count += 1
            
            numeric_values = [get_numeric_value(str(v)) for v in values]
            if all(v is not None for v in numeric_values):
                trend_info = calculate_trend(numeric_values)
                is_negative = is_negative_direction(question)
            else:
                trend_info = None
                is_negative = None
            
            hint = self._generate_prediction_hint(
                question, values, stability, trend_info, 
                sudden_shift, is_negative, consecutive_same
            )
            if hint:
                self.ltm_data['prediction_hints'][question] = hint
            
            self.ltm_data['temporal_patterns'][question] = {
                'topic': topic,
                'series': series,
                'stability': stability,
                'last_value': values[-1],
                'unique_count': len(unique_values),
                'mode': most_common_val,
                'mode_count': most_common_count,
                'mode_ratio': round(mode_ratio, 2),
                'total_years': len(values),
                'consecutive_same': consecutive_same,
                'trend': trend_info,
                'is_negative_direction': is_negative,
                'has_sudden_shift': sudden_shift is not None and sudden_shift.get('detected', False)
            }
        
        total_patterns = len(self.ltm_data['temporal_patterns'])
        self.ltm_data['data_quality']['sudden_shift_ratio'] = round(
            sudden_shift_count / total_patterns if total_patterns > 0 else 0, 3
        )
    
    def _generate_prediction_hint(self, question: str, values: List, 
                                   stability: str, trend: Optional[Dict],
                                   sudden_shift: Optional[Dict],
                                   is_negative: Optional[bool],
                                   consecutive_same: int) -> Dict:
        hints = []
        confidence = 'medium'
        suggested_value = values[-1]
        
        if stability == 'constant':
            hints.append(f"5년간 동일 응답 유지: '{values[-1]}'")
            confidence = 'very_high'
            suggested_value = values[-1]
        
        elif sudden_shift and sudden_shift.get('detected'):
            shift_year = sudden_shift['year']
            from_val = sudden_shift['from_value']
            to_val = sudden_shift['to_value']
            hints.append(f"주의: {shift_year}년 급변 ({from_val} -> {to_val})")
            hints.append("2023년도 변화 지속 또는 회귀 가능성 고려")
            confidence = 'low'
            suggested_value = to_val
        
        elif stability in ['highly_stable', 'stable']:
            if consecutive_same >= 3:
                hints.append(f"최근 {consecutive_same}년 연속 '{values[-1]}' 유지")
                confidence = 'high'
            else:
                hints.append(f"주된 응답: '{Counter(values).most_common(1)[0][0]}'")
                confidence = 'medium_high'
            suggested_value = values[-1]
        
        elif trend:
            direction = trend['direction']
            if direction == 'increasing':
                hints.append("상승 트렌드")
                if is_negative:
                    hints.append("(부정적 지표 악화 추세)")
            elif direction == 'decreasing':
                hints.append("하락 트렌드")
                if is_negative:
                    hints.append("(부정적 지표 개선 추세)")
            else:
                hints.append("안정적 추세")
            
            recent = trend.get('recent_change', 0)
            if abs(recent) > 0.5:
                hints.append(f"최근 변화폭: {'+' if recent > 0 else ''}{recent}")
            
            confidence = 'medium'
        
        return {
            'hints': hints,
            'confidence': confidence,
            'suggested_value': suggested_value,
            'basis': stability
        }
    
    def detect_consistency_issues(self):
        issues = []
        
        happiness_patterns = {}
        wellbeing_patterns = {}
        
        for q, data in self.ltm_data['temporal_patterns'].items():
            topic = data['topic']
            if topic == CATEGORY_HAPPINESS:
                happiness_patterns[q] = data
            elif topic in WELLBEING_CATEGORIES:
                wellbeing_patterns[q] = data
        
        for hq, hdata in happiness_patterns.items():
            h_val = get_numeric_value(str(hdata['last_value']))
            if h_val is None:
                continue
            
            for wq, wdata in wellbeing_patterns.items():
                w_topic = wdata['topic']
                
                if '불행' in str(hdata['last_value']):
                    if w_topic == CATEGORY_DEPRESSION:
                        w_numeric = get_numeric_value(str(wdata['last_value']))
                        if w_numeric and w_numeric <= 2:
                            issues.append({
                                'type': 'contradiction',
                                'description': f"행복도 낮음({hdata['last_value']}) vs 우울 낮음({wdata['last_value']})",
                                'questions': [hq, wq],
                                'severity': 'medium'
                            })
        
        topic_groups = defaultdict(list)
        for q, data in self.ltm_data['temporal_patterns'].items():
            topic_groups[data['topic']].append((q, data))
        
        for topic, items in topic_groups.items():
            if len(items) < 2:
                continue
            
            stabilities = [d['stability'] for _, d in items]
            if 'constant' in stabilities and 'variable' in stabilities:
                constant_qs = [q for q, d in items if d['stability'] == 'constant']
                variable_qs = [q for q, d in items if d['stability'] == 'variable']
                if constant_qs and variable_qs:
                    issues.append({
                        'type': 'inconsistent_stability',
                        'topic': topic,
                        'description': f"{topic} 내 안정성 불일치",
                        'constant_count': len(constant_qs),
                        'variable_count': len(variable_qs),
                        'severity': 'low'
                    })
        
        self.ltm_data['consistency_issues'] = issues
        self.ltm_data['data_quality']['consistency_issue_count'] = len(issues)
    
    def build_thematic_profiles(self):
        topic_questions = defaultdict(list)
        
        for question, data in self.ltm_data['temporal_patterns'].items():
            topic = data['topic']
            topic_questions[topic].append({
                'question': question,
                'stability': data['stability'],
                'last_value': data['last_value'],
                'mode': data.get('mode', ''),
                'mode_ratio': data.get('mode_ratio', 0),
                'has_sudden_shift': data.get('has_sudden_shift', False),
                'trend': data.get('trend')
            })
        
        for topic, questions in topic_questions.items():
            stability_counts = Counter(q['stability'] for q in questions)
            recent_values = [q['last_value'] for q in questions[:5]]
            
            stability_score = (
                stability_counts.get('constant', 0) * 1.0 +
                stability_counts.get('highly_stable', 0) * 0.8 +
                stability_counts.get('stable', 0) * 0.6 +
                stability_counts.get('variable', 0) * 0.3
            ) / len(questions) if questions else 0
            
            shift_count = sum(1 for q in questions if q['has_sudden_shift'])
            shift_ratio = shift_count / len(questions) if questions else 0
            
            trends = [q['trend'] for q in questions if q.get('trend')]
            if trends:
                directions = [t['direction'] for t in trends]
                dominant_direction = Counter(directions).most_common(1)[0][0]
            else:
                dominant_direction = 'unknown'
            
            self.ltm_data['thematic_profiles'][topic] = {
                'question_count': len(questions),
                'stability_distribution': dict(stability_counts),
                'stability_score': round(stability_score, 2),
                'sudden_shift_count': shift_count,
                'sudden_shift_ratio': round(shift_ratio, 2),
                'dominant_trend': dominant_direction,
                'sample_questions': [q['question'][:50] for q in questions[:3]],
                'recent_values_sample': recent_values
            }
    
    def calculate_data_quality(self):
        patterns = self.ltm_data['temporal_patterns']
        
        if not patterns:
            self.ltm_data['data_quality']['overall_score'] = 0
            return
        
        stability_scores = {
            'constant': 1.0, 'highly_stable': 0.8, 'stable': 0.6, 'variable': 0.3
        }
        avg_stability = sum(
            stability_scores.get(p['stability'], 0.5) for p in patterns.values()
        ) / len(patterns)
        
        shift_ratio = self.ltm_data['data_quality'].get('sudden_shift_ratio', 0)
        shift_penalty = shift_ratio * 0.3
        
        issue_count = self.ltm_data['data_quality'].get('consistency_issue_count', 0)
        consistency_penalty = min(issue_count * 0.05, 0.2)
        
        overall = max(0, avg_stability - shift_penalty - consistency_penalty)
        
        self.ltm_data['data_quality'].update({
            'overall_score': round(overall, 2),
            'stability_avg': round(avg_stability, 2),
            'shift_penalty': round(shift_penalty, 2),
            'consistency_penalty': round(consistency_penalty, 2),
            'prediction_difficulty': 'easy' if overall > 0.7 else 'medium' if overall > 0.5 else 'hard'
        })
    
    def build_yearly_changes(self):
        years = sorted([int(y) for y in self.history.keys()])
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            prev_data = self.history.get(str(prev_year), {})
            curr_data = self.history.get(str(curr_year), {})
            
            changes = []
            for q in curr_data.keys():
                if q in prev_data and prev_data[q] != curr_data[q]:
                    changes.append({
                        'question': q,
                        'topic': extract_category(q),
                        'before': prev_data[q],
                        'after': curr_data[q]
                    })
            
            if not changes:
                self.ltm_data['yearly_changes'][str(curr_year)] = {
                    'change_count': 0,
                    'summary': '변화없음',
                    'llm_analysis': '전년도와 동일한 응답 패턴'
                }
                continue
            
            topic_changes = defaultdict(list)
            for c in changes:
                topic_changes[c['topic']].append(c)
            
            changes_text = "\n".join([
                f"- [{c['topic']}] {c['question'][:30]}...: {c['before']} -> {c['after']}"
                for c in changes[:15]
            ])
            
            sudden_in_year = [
                q for q, shift in self.ltm_data['sudden_shifts'].items()
                if shift.get('year') == curr_year
            ]
            sudden_text = ""
            if sudden_in_year:
                sudden_text = f"\n\n[급격한 변화 감지 ({len(sudden_in_year)}개)]:\n"
                for q in sudden_in_year[:5]:
                    shift = self.ltm_data['sudden_shifts'][q]
                    sudden_text += f"- {q[:40]}...: {shift['from_value']} -> {shift['to_value']}\n"
            
            # Paper Figure 5: Yearly change analysis prompt
            prompt = f"""학생의 {prev_year}년에서 {curr_year}년 사이 변화를 분석하세요.

[주요 변화 ({len(changes)}개 항목)]
{changes_text}
{sudden_text}

다음 형식으로 분석하세요:
1. 핵심 변화 영역 (1-2개)
2. 변화의 방향성 (긍정적/부정적/혼재)
3. 주목할 패턴이나 우려 사항

3-4문장으로 요약:"""
            
            try:
                response = self.llm.invoke(prompt)
                llm_analysis = response.content.strip()
            except Exception as e:
                llm_analysis = f"분석 실패: {str(e)[:50]}"
            
            self.ltm_data['yearly_changes'][str(curr_year)] = {
                'change_count': len(changes),
                'by_topic': {t: len(cs) for t, cs in topic_changes.items()},
                'key_changes': changes[:5],
                'sudden_shifts_count': len(sudden_in_year),
                'llm_analysis': llm_analysis
            }
    
    def build_category_analysis(self):
        """
        Paper Figure 7: Category-level analysis prompt.
        Analyzes category summaries, cross-category correlations, and sudden changes.
        """
        # Build category summaries
        category_summaries = []
        for topic, profile in sorted(self.ltm_data['thematic_profiles'].items()):
            q_count = profile.get('question_count', 0)
            stability = profile.get('stability_score', 0)
            trend = profile.get('dominant_trend', 'unknown')
            shift_ratio = profile.get('sudden_shift_ratio', 0)
            recent_vals = profile.get('recent_values_sample', [])
            
            summary_parts = [f"- [{topic}]"]
            if shift_ratio > 0:
                summary_parts.append(f"  안정성: {stability:.2f}, 급변율: {shift_ratio:.0%}, 트렌드: {trend}")
            else:
                summary_parts.append(f"  안정성: {stability:.2f}, 트렌드: {trend}")
            if recent_vals:
                summary_parts.append(f"  최근 응답: {', '.join(str(v) for v in recent_vals[:3])}")
            
            category_summaries.append('\n'.join(summary_parts))
        
        # Build cross-category correlations
        strong_correlations = []
        topics = list(self.ltm_data['thematic_profiles'].keys())
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                profile1 = self.ltm_data['thematic_profiles'].get(topic1, {})
                profile2 = self.ltm_data['thematic_profiles'].get(topic2, {})
                
                trend1 = profile1.get('dominant_trend', '')
                trend2 = profile2.get('dominant_trend', '')
                
                if trend1 == trend2 and trend1 in ['increasing', 'decreasing']:
                    strong_correlations.append(f"- '{topic1}' <-> '{topic2}': 동일 트렌드 ({trend1})")
                elif (trend1 == 'increasing' and trend2 == 'decreasing') or \
                     (trend1 == 'decreasing' and trend2 == 'increasing'):
                    strong_correlations.append(f"- '{topic1}' <-> '{topic2}': 반대 트렌드")
        
        # Build sudden changes list
        sudden_changes = []
        last_year = max([int(y) for y in self.history.keys()]) if self.history else 2022
        
        for q, shift in self.ltm_data['sudden_shifts'].items():
            if shift.get('year') == last_year:
                topic = extract_category(q)
                from_val = shift.get('from_value', '')
                to_val = shift.get('to_value', '')
                z_score = shift.get('z_score', '')
                
                change_desc = f"- [{last_year-1}->{last_year}] '{topic}': {from_val} -> {to_val}"
                if z_score:
                    change_desc += f" (Z-score: {z_score})"
                sudden_changes.append(change_desc)
        
        # Paper Figure 7: Category analysis prompt
        prompt = f"""학생 {self.student_id}의 설문 데이터를 분석하세요.

[카테고리별 현황]
{chr(10).join(category_summaries)}

[카테고리 간 상관관계]
{chr(10).join(strong_correlations[:5]) if strong_correlations else '특이 상관 없음'}

[{last_year}년 급변 문항 ({len(sudden_changes)}개)]
{chr(10).join(sudden_changes[:5]) if sudden_changes else '없음'}

다음을 분석하세요:
1. 전반적인 발달 특성
2. 카테고리 간 연관 패턴
3. 2023년 예측 시 고려사항

3-4문장으로 요약:"""
        
        try:
            response = self.llm.invoke(prompt)
            self.ltm_data['category_analysis'] = {
                'summary': response.content.strip(),
                'category_count': len(category_summaries),
                'correlation_count': len(strong_correlations),
                'sudden_change_count': len(sudden_changes)
            }
        except Exception as e:
            self.ltm_data['category_analysis'] = {
                'summary': f"분석 실패: {str(e)[:50]}",
                'category_count': len(category_summaries),
                'correlation_count': len(strong_correlations),
                'sudden_change_count': len(sudden_changes)
            }
    
    def build_overall_narrative(self):
        """Paper Figure 6: Overall narrative prompt."""
        yearly_analyses = []
        for year, data in sorted(self.ltm_data['yearly_changes'].items()):
            analysis = data.get('llm_analysis', '')
            if analysis and '분석 실패' not in analysis:
                yearly_analyses.append(f"[{year}년] {analysis[:Config.YEARLY_SUMMARY_MAX_LENGTH]}")
        
        quality = self.ltm_data['data_quality']
        difficulty = quality.get('prediction_difficulty', 'unknown')
        shift_ratio = quality.get('sudden_shift_ratio', 0)
        
        topic_summary = []
        for topic, profile in sorted(
            self.ltm_data['thematic_profiles'].items(),
            key=lambda x: x[1].get('sudden_shift_ratio', 0),
            reverse=True
        )[:5]:
            score = profile.get('stability_score', 0)
            shift_r = profile.get('sudden_shift_ratio', 0)
            trend = profile.get('dominant_trend', 'unknown')
            topic_summary.append(f"- {topic}: 안정성={score:.1f}, 급변율={shift_r:.0%}, 트렌드={trend}")
        
        shift_topics = Counter(
            self.ltm_data['temporal_patterns'].get(q, {}).get('topic', 'unknown')
            for q in self.ltm_data['sudden_shifts'].keys()
        )
        top_shift_topics = shift_topics.most_common(3)
        
        # Paper Figure 6: Overall narrative prompt
        prompt = f"""학생 {self.student_id}의 5년간(2018-2022) 종단 데이터를 종합 분석하세요.

[연도별 변화 분석]
{chr(10).join(yearly_analyses) if yearly_analyses else '특이 변화 없음'}

[데이터 품질]
- 예측 난이도: {difficulty}
- 급격한 변화 비율: {shift_ratio:.1%}
- 급변 집중 영역: {', '.join([f'{t}({c}개)' for t, c in top_shift_topics]) if top_shift_topics else '없음'}

[주제별 특성]
{chr(10).join(topic_summary)}

다음을 포함하여 종합적인 성장 서사를 작성하세요:
1. 전반적인 발달 궤적과 특징
2. 주요 전환점 또는 급변 시점
3. 강점 영역과 우려 영역
4. 2023년 예측 시 반드시 고려해야 할 점

5-7문장으로 작성:"""
        
        try:
            response = self.llm.invoke(prompt)
            self.ltm_data['overall_narrative'] = response.content.strip()
        except Exception as e:
            narrative_parts = [f"예측 난이도: {difficulty}"]
            if top_shift_topics:
                narrative_parts.append(f"2022년 주요 변화 영역: {top_shift_topics[0][0]} ({top_shift_topics[0][1]}개 문항)")
            stable_topics = [t for t, p in self.ltm_data['thematic_profiles'].items() if p.get('stability_score', 0) > 0.7]
            if stable_topics:
                narrative_parts.append(f"안정적 영역: {', '.join(stable_topics[:3])}")
            self.ltm_data['overall_narrative'] = " | ".join(narrative_parts) + f" (LLM 실패: {str(e)[:30]})"
    
    def build_all(self) -> Dict[str, Any]:
        if not self.history:
            return self.ltm_data
        
        self.build_static_persona()
        
        self.build_temporal_patterns()
        self.detect_consistency_issues()
        self.build_thematic_profiles()
        self.calculate_data_quality()
        
        self.build_yearly_changes()
        self.build_category_analysis()  # Paper Figure 7
        self.build_overall_narrative()  # Paper Figure 6
        
        self.ltm_data['metadata']['build_time'] = datetime.now().isoformat()
        return self.ltm_data
    
    def save(self, output_dir: str = None, suffix: str = "full_pipeline") -> str:
        if output_dir is None:
            output_dir = Config.MIRROR_MEMORY_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.student_id}_ltm_{suffix}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.ltm_data, f, ensure_ascii=False, indent=2)
        
        return output_path


def build_ltm_for_student(student_id: str, llm: ChatOllama = None, 
                          output_dir: str = None) -> Tuple[str, float, Dict]:
    start = time.time()
    
    builder = LTMBuilder(student_id, llm=llm)
    builder.build_all()
    output_path = builder.save(output_dir)
    
    elapsed = time.time() - start
    
    stats = {
        'patterns': len(builder.ltm_data['temporal_patterns']),
        'topics': len(builder.ltm_data['thematic_profiles']),
        'sudden_shifts': len(builder.ltm_data['sudden_shifts']),
        'prediction_difficulty': builder.ltm_data['data_quality'].get('prediction_difficulty', 'unknown')
    }
    
    return output_path, elapsed, stats


def build_ltm_batch(student_ids: List[str], output_dir: str = None, 
                    max_workers: int = 2, skip_existing: bool = False):
    print(f"Building LTM for {len(student_ids)} students...")
    print(f"Workers: {max_workers}, Skip existing: {skip_existing}")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = Config.MIRROR_MEMORY_DIR
    
    if skip_existing:
        existing = set()
        for f in os.listdir(output_dir) if os.path.exists(output_dir) else []:
            if f.endswith('_ltm_full_pipeline.json'):
                sid = f.replace('_ltm_full_pipeline.json', '')
                existing.add(sid)
        
        original_count = len(student_ids)
        student_ids = [sid for sid in student_ids if sid not in existing]
        print(f"Skipping {original_count - len(student_ids)} existing files")
    
    if not student_ids:
        print("No students to process")
        return []
    
    results = []
    total_time = 0
    errors = []
    
    def process_student(sid):
        llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=800,
            num_ctx=Config.NUM_CTX,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
        return sid, build_ltm_for_student(sid, llm=llm, output_dir=output_dir)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_student, sid): sid for sid in student_ids}
        
        with tqdm(total=len(student_ids), desc="Building LTM") as pbar:
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    _, (path, elapsed, stats) = future.result()
                    results.append((sid, True, elapsed, stats))
                    total_time += elapsed
                    pbar.set_postfix({
                        'last': f"{sid}({elapsed:.1f}s)",
                        'difficulty': stats['prediction_difficulty'],
                        'shifts': stats['sudden_shifts']
                    })
                except Exception as e:
                    results.append((sid, False, 0, str(e)))
                    errors.append((sid, str(e)))
                pbar.update(1)
    
    success = sum(1 for r in results if r[1])
    failed = len(results) - success
    
    print("=" * 60)
    print(f"Complete: {success} success, {failed} failed")
    if total_time > 0:
        print(f"Total time: {total_time/60:.1f} min")
        print(f"Avg per student: {total_time/max(success,1):.1f}s")
    
    if results:
        success_results = [r for r in results if r[1]]
        if success_results:
            difficulties = Counter(r[3]['prediction_difficulty'] for r in success_results)
            print(f"Prediction difficulty: {dict(difficulties)}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for sid, err in errors[:5]:
            print(f"  - {sid}: {err[:50]}")
    
    return results


def load_student_ltm(student_id: str, ltm_dir: str = None) -> Optional[Dict]:
    if ltm_dir is None:
        ltm_dir = Config.MIRROR_MEMORY_DIR
    
    for suffix in ['full_pipeline', 'rich']:
        ltm_path = os.path.join(ltm_dir, f"{student_id}_ltm_{suffix}.json")
        if os.path.exists(ltm_path):
            with open(ltm_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LTM for students")
    parser.add_argument("--student", type=str, help="Single student ID")
    parser.add_argument("--all", action="store_true", help="Build for all students")
    parser.add_argument("--student-range", type=int, nargs=2, metavar=('START', 'END'))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=Config.LTM_WORKERS if hasattr(Config, 'LTM_WORKERS') else 2,
                        help="Parallel workers (default: 2)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    
    if args.all or args.student_range:
        all_ids = sorted(get_all_student_ids(Config.DATA_DIR))
        
        if args.student_range:
            start, end = args.student_range
            start = max(0, start)
            end = min(len(all_ids) - 1, end)
            student_ids = all_ids[start:end + 1]
            print(f"[Range Mode] index {start} to {end} ({len(student_ids)} students)")
        else:
            student_ids = all_ids
        
        build_ltm_batch(
            student_ids, 
            output_dir=args.output_dir, 
            max_workers=args.workers,
            skip_existing=args.skip_existing
        )
        
    elif args.student:
        path, elapsed, stats = build_ltm_for_student(args.student, output_dir=args.output_dir)
        print(f"[{args.student}] Built in {elapsed:.1f}s")
        print(f"  patterns: {stats['patterns']}, shifts: {stats['sudden_shifts']}")
        print(f"  difficulty: {stats['prediction_difficulty']}")
        print(f"Saved to: {path}")
    else:
        print("Usage:")
        print("  python build_ltm.py --student 10001")
        print("  python build_ltm.py --all --workers 2")
        print("  python build_ltm.py --all --skip-existing")
