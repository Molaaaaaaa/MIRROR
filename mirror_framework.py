"""
MIRROR Framework Core Module (RAG + LLM Deep Reasoning)

Core Components:
1. Related category search: Uses RAG/KG (fast)
2. Prediction: LLM-based deep reasoning (LTM narrative + related context + sudden shifts)
3. Option type recognition by category (frequency vs agreement level)
"""
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from langchain_ollama import ChatOllama
from data_constants import (
    FREQUENCY_KEYWORDS,
    AGREEMENT_KEYWORDS,
    LOW_LEVEL_KEYWORDS,
    HIGH_LEVEL_KEYWORDS,
)

from config import Config
from tools import MirrorToolkit, TOOL_SETS
from utils import load_student_data, clean_llm_output, get_fallback_prediction, extract_category


# Category embedding similarity cache
_CATEGORY_SIMILARITY_CACHE = None

def _load_category_similarity() -> Dict:
    """Load category embedding similarity"""
    global _CATEGORY_SIMILARITY_CACHE
    if _CATEGORY_SIMILARITY_CACHE is None:
        similarity_path = os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
        if os.path.exists(similarity_path):
            with open(similarity_path, 'r', encoding='utf-8') as f:
                _CATEGORY_SIMILARITY_CACHE = json.load(f)
        else:
            _CATEGORY_SIMILARITY_CACHE = {}
    return _CATEGORY_SIMILARITY_CACHE


# Behavioral correlation cache (all students based)
_BEHAVIORAL_CORRELATION_CACHE = None

def _load_behavioral_correlation() -> Dict:
    global _BEHAVIORAL_CORRELATION_CACHE
    if _BEHAVIORAL_CORRELATION_CACHE is None:
        corr_path = os.path.join(Config.MIRROR_MEMORY_DIR, "behavioral_correlation.json")
        if os.path.exists(corr_path):
            with open(corr_path, 'r', encoding='utf-8') as f:
                _BEHAVIORAL_CORRELATION_CACHE = json.load(f)
        else:
            _BEHAVIORAL_CORRELATION_CACHE = {}
    return _BEHAVIORAL_CORRELATION_CACHE

def detect_option_type(options: Dict[str, str]) -> str:
    """Detect option type (frequency vs agreement level)"""
    option_text = ' '.join(options.values())
    
    freq_score = sum(1 for kw in FREQUENCY_KEYWORDS if kw in option_text)
    agree_score = sum(1 for kw in AGREEMENT_KEYWORDS if kw in option_text)
    
    if freq_score > agree_score:
        return 'frequency'
    elif agree_score > 0:
        return 'agreement'
    else:
        return 'other'


def get_option_type_description(option_type: str) -> str:
    """Get description for option type"""
    if option_type == 'frequency':
        return "발생 빈도를 선택하세요."
    elif option_type == 'agreement':
        return "동의 수준을 선택하세요."
    else:
        return ""


class MirrorPredictor:
    def __init__(self, student_id: str, exclude_target: bool = False, 
                 exclude_partial: bool = False, tool_set: str = 'full',
                 debug: bool = False):
        self.student_id = student_id
        self.tool_set = tool_set
        self.debug = debug
        self.enabled_tools = TOOL_SETS.get(tool_set, TOOL_SETS['full'])
        
        _, self.history = load_student_data(
            Config.DATA_DIR, student_id, 
            exclude_target=exclude_target,
            exclude_partial=exclude_partial
        )
        
        self.ltm_data = self._load_data('ltm')
        self.kg_data = self._load_data('kg') if 'kg_query' in self.enabled_tools else {}
        
        self.toolkit = MirrorToolkit(
            student_id=student_id,
            history=self.history,
            ltm_data=self.ltm_data,
            kg_data=self.kg_data,
            tool_set=tool_set
        )
        
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=500,
            num_ctx=4096,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
        
        self._category_profiles = None
        self._available_categories = None
        self._related_categories_cache = {}
        
        # LTM narrative cache
        self._narrative_cache = None
        
        if self.debug:
            print(f"[DEBUG] LTM: {len(self.ltm_data.get('temporal_patterns', {}))} patterns")
    
    def _load_data(self, data_type: str) -> Dict:
        if data_type == 'ltm':
            for suffix in ['full_pipeline', 'rich', '']:
                filename = f"{self.student_id}_ltm_{suffix}.json" if suffix else f"{self.student_id}_ltm.json"
                path = os.path.join(Config.MIRROR_MEMORY_DIR, filename)
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        else:
            path = os.path.join(Config.MIRROR_MEMORY_DIR, f"{self.student_id}_kg.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return {}
    
    def _get_narrative(self) -> str:
        """Get LTM narrative (cached)"""
        if self._narrative_cache is None:
            self._narrative_cache = self.ltm_data.get('overall_narrative', '')
        return self._narrative_cache
    
    def _normalize_for_match(self, text: str) -> str:
        text = re.sub(r'\s+', '', text)
        return text.lower()
    
    def _find_in_dict(self, question: str, data_dict: Dict) -> Optional[Any]:
        if not data_dict:
            return None
        
        if question in data_dict:
            return data_dict[question]
        
        norm_q = self._normalize_for_match(question)
        for key, value in data_dict.items():
            if self._normalize_for_match(key) == norm_q:
                return value
        
        q_core = question.split(']')[-1].strip() if ']' in question else question
        q_core_norm = self._normalize_for_match(q_core)
        
        for key, value in data_dict.items():
            key_core = key.split(']')[-1].strip() if ']' in key else key
            if self._normalize_for_match(key_core) == q_core_norm:
                return value
        
        return None
    
    def _get_temporal_pattern(self, question: str) -> Optional[Dict]:
        return self._find_in_dict(question, self.ltm_data.get('temporal_patterns', {}))
    
    def _get_prediction_hint(self, question: str) -> Optional[Dict]:
        return self._find_in_dict(question, self.ltm_data.get('prediction_hints', {}))
    
    def _get_sudden_shift(self, question: str) -> Optional[Dict]:
        return self._find_in_dict(question, self.ltm_data.get('sudden_shifts', {}))
    
    def _get_question_history(self, question: str) -> List[Dict]:
        history = []
        norm_q = self._normalize_for_match(question)
        q_core = question.split(']')[-1].strip() if ']' in question else question
        q_core_norm = self._normalize_for_match(q_core)
        
        for year in sorted(self.history.keys()):
            if int(year) >= Config.TARGET_YEAR:
                continue
            
            if question in self.history[year]:
                history.append({'year': int(year), 'value': self.history[year][question]})
                continue
            
            for q, v in self.history[year].items():
                if self._normalize_for_match(q) == norm_q:
                    history.append({'year': int(year), 'value': v})
                    break
                q_c = q.split(']')[-1].strip() if ']' in q else q
                if self._normalize_for_match(q_c) == q_core_norm:
                    history.append({'year': int(year), 'value': v})
                    break
        
        return history
    
    def _find_option_key(self, value: str, options: Dict[str, str]) -> str:
        if not value or not options:
            return "0"
        
        value_norm = self._normalize_for_match(value)
        
        for k, v in options.items():
            if v == value:
                return k
        
        for k, v in options.items():
            if self._normalize_for_match(v) == value_norm:
                return k
        
        for k, v in options.items():
            v_norm = self._normalize_for_match(v)
            if value_norm in v_norm or v_norm in value_norm:
                return k
        
        return "0"
    
    def _get_response_level_from_pattern(self, pattern: Dict) -> str:
        """
        determine response level (low/mid/high) from temporal pattern
        """
        trend = pattern.get('trend') if pattern else None
        
        if trend and isinstance(trend, dict):
            last_val_numeric = trend.get('last_value')
            
            if last_val_numeric is not None:
                if last_val_numeric <= 2:
                    return "low"
                elif last_val_numeric <= 3:
                    return "mid"
                else:
                    return "high"
        
        last_value = pattern.get('last_value', '') if pattern else ''
        return self._infer_level_from_text(last_value)
    
    def _infer_level_from_text(self, value: str) -> str:
        """estimate response level from text value"""
        if not value:
            return "mid"
        
        low_keywords = LOW_LEVEL_KEYWORDS
        high_keywords = HIGH_LEVEL_KEYWORDS
        
        for kw in low_keywords:
            if kw in value:
                return "low"
        for kw in high_keywords:
            if kw in value:
                return "high"
        return "mid"
    
    def _get_pattern_description(self, pattern: Dict) -> str:
        """
        describe temporal pattern
        """
        if not pattern:
            return "알 수 없음"
        
        stability = pattern.get('stability', 'unknown')
        level = self._get_response_level_from_pattern(pattern)
        last_value = pattern.get('last_value', '')
        
        trend = pattern.get('trend')
        direction = ''
        if trend and isinstance(trend, dict):
            direction = trend.get('direction', '')
        
        stab_desc = {
            'constant': '일정',
            'highly_stable': '매우안정',
            'stable': '안정',
            'variable': '변동'
        }.get(stability, stability)
        
        dir_desc = ''
        if direction == 'increasing':
            dir_desc = ', 증가 추세'
        elif direction == 'decreasing':
            dir_desc = ', 감소 추세'
        
        return f"{stab_desc}-{level}{dir_desc} (최근값: '{last_value}')"
    
    def _get_correlation_info(self, target_category: str, related_category: str) -> str:
        """
        return correlation info between two categories
        """
        behavioral_corr = _load_behavioral_correlation()
        if not behavioral_corr:
            return ""
        
        relationships = behavioral_corr.get('category_relationships', {})
        
        corr_value = None
        if target_category in relationships:
            for item in relationships[target_category]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if item[0] == related_category:
                        corr_value = item[1]
                        break
        
        if corr_value is None and related_category in relationships:
            for item in relationships[related_category]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if item[0] == target_category:
                        corr_value = item[1]
                        break
        
        if corr_value is None:
            return ""
        
        sign = "+" if corr_value > 0 else ""
        strength = abs(corr_value)
        
        if strength >= 0.5:
            strength_desc = "강한"
        elif strength >= 0.3:
            strength_desc = "중간"
        else:
            strength_desc = "약한"
        
        direction_desc = "양의" if corr_value > 0 else "음의"
        
        return f"(상관: {sign}{corr_value:.2f}, {strength_desc} {direction_desc} 관계)"
    
    def _find_related_categories_by_kg(self, category: str) -> Tuple[List[str], Dict[str, float]]:
        related = []
        scores = {}
        
        behavioral_corr = _load_behavioral_correlation()
        if behavioral_corr:
            relationships = behavioral_corr.get('category_relationships', {})
            if category in relationships:
                for item in relationships[category][:5]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        cat_name, corr = item[0], item[1]
                        related.append(cat_name)
                        scores[cat_name] = corr
                    elif isinstance(item, (list, tuple)) and len(item) >= 1:
                        related.append(item[0])
                        scores[item[0]] = 0.5
                
                if self.debug and related:
                    print(f"[DEBUG] Behavioral correlation for '{category}': {related[:3]}")
        
        if not related:
            category_sim = _load_category_similarity()
            if category_sim:
                sim_relationships = category_sim.get('category_relationships', {})
                if category in sim_relationships:
                    for item in sim_relationships[category][:5]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            cat_name, score = item[0], item[1]
                            related.append(cat_name)
                            scores[cat_name] = score * 0.8
                        elif isinstance(item, (list, tuple)) and len(item) >= 1:
                            related.append(item[0])
                            scores[item[0]] = 0.4
                    
                    if self.debug and related:
                        print(f"[DEBUG] Embedding similarity fallback for '{category}': {related[:3]}")
        
        return related, scores
    
    def _find_related_categories_by_rag(self, question: str, category: str) -> List[str]:
        """find related categories using RAG search"""
        if 'rag_search' not in self.enabled_tools or not self.toolkit.vectorstore:
            return []
        
        try:
            docs = self.toolkit.vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(question)
            
            related_cats = []
            for doc in docs:
                doc_cat = doc.metadata.get('category', '')
                if doc_cat and doc_cat != category and doc_cat not in related_cats:
                    related_cats.append(doc_cat)
            
            return related_cats[:5]
        except:
            return []
    
    def _find_related_categories_dynamic(self, question: str, category: str) -> List[str]:
        """dynamically find related categories (with caching)"""
        if category in self._related_categories_cache:
            return self._related_categories_cache[category]
        
        behavioral_corr = _load_behavioral_correlation()
        has_behavioral = category in behavioral_corr.get('category_relationships', {})
        
        if not has_behavioral:
            category_sim = _load_category_similarity()
            sim_rels = category_sim.get('category_relationships', {}).get(category, [])
            if sim_rels:
                top_related = [item[0] for item in sim_rels[:3] if isinstance(item, (list, tuple)) and len(item) >= 1]
                
                if self.debug:
                    debug_info = [(item[0], item[1] if len(item) > 1 else 0) for item in sim_rels[:3]]
                    print(f"[DEBUG] New category '{category}' -> similarity top-3: {debug_info}")
                
                self._related_categories_cache[category] = top_related
                return top_related
        
        vote_counter = Counter()
        score_sum = {}
        sources = {}
        
        use_behavioral = 'kg_query' in self.enabled_tools
        
        if use_behavioral:
            behavioral_cats, behavioral_scores = self._find_related_categories_by_kg(category)
            for cat in behavioral_cats:
                if cat != category:
                    vote_counter[cat] += 1
                    score_sum[cat] = score_sum.get(cat, 0) + abs(behavioral_scores.get(cat, 0.5))
                    sources[cat] = sources.get(cat, []) + ['BEH']
        
        rag_cats = self._find_related_categories_by_rag(question, category)
        for cat in rag_cats[:3]:
            if cat != category:
                vote_counter[cat] += 1
                score_sum[cat] = score_sum.get(cat, 0) + 0.4
                sources[cat] = sources.get(cat, []) + ['RAG']
        
        candidates = list(vote_counter.keys())
        candidates.sort(key=lambda x: (vote_counter[x], score_sum.get(x, 0)), reverse=True)
        
        result = candidates[:3]
        
        if self.debug and result:
            debug_info = [f"{cat}({vote_counter[cat]}votes,{score_sum.get(cat,0):.2f},{'+'.join(sources.get(cat,[]))})" for cat in result]
            print(f"[DEBUG] Related for '{category}': {debug_info}")
        
        self._related_categories_cache[category] = result
        return result
    
    def _build_category_profiles(self) -> Dict[str, Dict]:
        if self._category_profiles is not None:
            return self._category_profiles
        
        profiles = {}
        patterns = self.ltm_data.get('temporal_patterns', {})
        
        for q, pattern in patterns.items():
            category = pattern.get('topic', extract_category(q))
            
            if category not in profiles:
                profiles[category] = {
                    'last_values': [], 
                    'stability_scores': [],
                    'total': 0
                }
            
            prof = profiles[category]
            prof['total'] += 1
            
            last_val = pattern.get('last_value', '')
            if last_val:
                prof['last_values'].append(last_val)
            
            stability = pattern.get('stability', '')
            if stability == 'constant':
                prof['stability_scores'].append(1.0)
            elif stability == 'highly_stable':
                prof['stability_scores'].append(0.8)
            elif stability == 'stable':
                prof['stability_scores'].append(0.6)
            else:
                prof['stability_scores'].append(0.3)
        
        for cat, prof in profiles.items():
            if prof['stability_scores']:
                prof['avg_stability'] = sum(prof['stability_scores']) / len(prof['stability_scores'])
            else:
                prof['avg_stability'] = 0.5
        
        self._category_profiles = profiles
        return profiles
    
    def _get_related_context(self, related_cats: List[str], target_category: str = "") -> str:
        patterns = self.ltm_data.get('temporal_patterns', {})
        context_parts = []
        
        for cat in related_cats[:3]:
            corr_info = ""
            if target_category:
                corr_info = self._get_correlation_info(target_category, cat)
            
            cat_items = []
            for q, pattern in patterns.items():
                if pattern.get('topic') == cat:
                    q_short = q.split(']')[-1].strip() if ']' in q else q
                    q_short = q_short[:25] + "..." if len(q_short) > 25 else q_short
                    
                    last_val = pattern.get('last_value', '')
                    stability = pattern.get('stability', '')
                    level = self._get_response_level_from_pattern(pattern)
                    
                    cat_items.append(f"  - {q_short}: {last_val} ({stability}, {level})")
            
            if cat_items:
                header = f"[{cat}]"
                if corr_info:
                    header += f" {corr_info}"
                context_parts.append(header + "\n" + "\n".join(cat_items[:5]))
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _predict_new_question(self, question: str, options: Dict[str, str], category: str) -> Tuple[str, str, Dict]:
        ctx = {'is_new_question': True, 'category': category}
        
        related_cats = self._find_related_categories_dynamic(question, category)
        ctx['related_categories'] = related_cats
        
        related_context = self._get_related_context(related_cats, category)
        
        narrative = self._get_narrative()
        narrative_text = narrative[:Config.NARRATIVE_MAX_LENGTH] if narrative else '정보 없음'
        
        option_type = detect_option_type(options)
        option_type_desc = get_option_type_description(option_type)
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        prompt = f"""첫 번째 줄에 답변 번호만 출력하세요.
예시:
Answer: 2

이 학생의 2018-2022 종단 데이터를 분석하여 2023년 응답을 예측하세요.

[학생 성장 서사]
{narrative_text}

[관련 카테고리 분석]
{related_context if related_context else '관련 정보 없음'}

[예측 대상]
카테고리: {category}
질문: {question}

[선택지]
{options_str}
({option_type_desc})

[지침]
1. 관련 카테고리의 응답 패턴과 수준(low/mid/high)을 고려하세요
2. 상관관계 방향을 고려하세요 (양의 상관: 유사한 수준, 음의 상관: 반대 수준)
3. 학생의 전반적인 성향과 발달 궤적을 고려하세요

숫자만 출력하세요.
Answer:"""

        try:
            response = self.llm.invoke(prompt)
            raw_output = response.content
            pred = clean_llm_output(raw_output)
            
            if self.debug:
                print(f"[DEBUG] NewQ LLM Raw: '{raw_output}'")
                print(f"[DEBUG] NewQ Cleaned: '{pred}', Valid options: {list(options.keys())}")
            
            if pred != "0" and pred in options:
                return pred, f"NewQ:LLM:{','.join(related_cats[:2]) if related_cats else 'none'}", ctx
            
            if self.debug:
                print(f"[DEBUG] NewQ Fallback: pred='{pred}' not valid")
                
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM Exception: {type(e).__name__}: {e}")
        
        option_keys = sorted(options.keys())
        return option_keys[0] if option_keys else "1", "NewQ:Fallback", ctx
    
    def _predict_existing_question(self, question: str, options: Dict[str, str], 
                                    ctx: Dict) -> Tuple[str, str, Dict]:
        
        category = extract_category(question)
        pattern = self._get_temporal_pattern(question)
        
        hist = self._get_question_history(question)
        if hist:
            trace = " -> ".join([f"{h['year']}:{h['value']}" for h in hist])
        else:
            trace = f"최근값: {ctx.get('last_value', '')}" if ctx.get('last_value') else "정보 없음"
        
        pattern_desc = ""
        if pattern:
            pattern_desc = self._get_pattern_description(pattern)
        else:
            stability = ctx.get('stability', '')
            last_value = ctx.get('last_value', '')
            pattern_desc = f"{stability} (최근값: '{last_value}')"
        
        shift = self._get_sudden_shift(question)
        shift_warning = ""
        if shift:
            shift_year = shift.get('year', '')
            from_val = shift.get('from_value', '')
            to_val = shift.get('to_value', '')
            shift_warning = f"\n[주의] {shift_year}년 급변 감지: '{from_val}' -> '{to_val}'"
        
        narrative = self._get_narrative()
        narrative_short = narrative[:Config.NARRATIVE_SHORT_LENGTH] if narrative else ''
        
        related_cats = self._find_related_categories_dynamic(question, category)
        related_context = self._get_related_context(related_cats[:3], category)
        
        option_type = detect_option_type(options)
        option_type_desc = get_option_type_description(option_type)
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        prompt = f"""첫 번째 줄에 답변 번호만 출력하세요.
예시:
Answer: 2

이 학생의 시계열 데이터를 분석하여 2023년 응답을 예측하세요.

[학생 개요]
{narrative_short if narrative_short else '정보 없음'}

[시계열 (2018-2022)]
{trace}
패턴: {pattern_desc}
{shift_warning}

[관련 카테고리]
{related_context if related_context else ''}

[예측 대상]
질문: {question}

[선택지]
{options_str}
({option_type_desc})

[지침]
1. 패턴 안정성과 응답 수준(low/mid/high)을 고려하세요
2. 급변이 있는 경우 지속 vs 회귀를 판단하세요
3. 관련 카테고리와의 상관관계를 고려하세요

숫자만 출력하세요.
Answer:"""

        try:
            response = self.llm.invoke(prompt)
            raw_output = response.content
            pred = clean_llm_output(raw_output)
            
            if self.debug:
                print(f"[DEBUG] Existing LLM Raw: '{raw_output[:300]}'")
                print(f"[DEBUG] Existing Cleaned: '{pred}', Valid options: {list(options.keys())}")
            
            if pred != "0" and pred in options:
                method = f"LLM:{ctx.get('stability', '')}"
                if shift:
                    method += ":shift"
                return pred, method, ctx
                
            if self.debug:
                print(f"[DEBUG] Existing Fallback: pred='{pred}' not valid")
                
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM Exception: {type(e).__name__}: {e}")
        
        if ctx.get('suggested_value'):
            pred_key = self._find_option_key(ctx['suggested_value'], options)
            if pred_key != "0":
                return pred_key, "Fallback:Suggested", ctx
        
        if ctx.get('last_value'):
            pred_key = self._find_option_key(ctx['last_value'], options)
            if pred_key != "0":
                return pred_key, "Fallback:LastValue", ctx
        
        hist = self._get_question_history(question)
        pred = get_fallback_prediction(hist, options)
        return pred, "Fallback:Final", ctx
    
    def _build_structured_context(self, question: str) -> Dict:
        context = {
            'stability': None,
            'last_value': None,
            'suggested_value': None,
            'confidence': None,
            'has_sudden_shift': False,
            'is_new_question': False
        }
        
        pattern = self._get_temporal_pattern(question)
        if pattern:
            context['stability'] = pattern.get('stability', 'unknown')
            context['last_value'] = pattern.get('last_value', '')
            context['has_sudden_shift'] = pattern.get('has_sudden_shift', False)
        
        hint = self._get_prediction_hint(question)
        if hint:
            context['suggested_value'] = hint.get('suggested_value', '')
            context['confidence'] = hint.get('confidence', 'medium')
        
        shift = self._get_sudden_shift(question)
        if shift:
            context['has_sudden_shift'] = True
        
        if not pattern:
            hist = self._get_question_history(question)
            if hist:
                context['last_value'] = hist[-1]['value'] if hist else ''
                
                values = [h['value'] for h in hist]
                unique = len(set(values))
                if unique == 1:
                    context['stability'] = 'constant'
                    context['confidence'] = 'very_high'
                    context['suggested_value'] = values[-1]
                elif unique <= 2 and len(values) >= 3:
                    counter = Counter(values)
                    mode_val, mode_count = counter.most_common(1)[0]
                    if mode_count >= len(values) * 0.6:
                        context['stability'] = 'highly_stable'
                        context['confidence'] = 'high'
                        context['suggested_value'] = values[-1]
                else:
                    context['stability'] = 'variable'
            else:
                context['is_new_question'] = True
        
        return context
    
    def predict(self, question: str, options: Dict[str, str]) -> Tuple[str, str, Dict]:
        category = extract_category(question)
        ctx = self._build_structured_context(question)
        
        if self.debug:
            print(f"\n[DEBUG] Q: {question[:50]}...")
            print(f"[DEBUG] Cat: {category}, Stability: {ctx.get('stability')}, Shift: {ctx.get('has_sudden_shift')}")
            print(f"[DEBUG] Options: {options}")
        
        if ctx['is_new_question']:
            return self._predict_new_question(question, options, category)
        
        return self._predict_existing_question(question, options, ctx)
    
    def predict_batch(self, tasks: List[Dict], verbose: bool = False) -> Tuple[List[str], List[str]]:
        from tqdm import tqdm
        preds, reasons = [], []
        
        for task in tqdm(tasks, desc=f"[{self.tool_set}] {self.student_id}"):
            pred, reason, _ = self.predict(task['question'], task['options'])
            preds.append(pred)
            reasons.append(reason)
        
        return preds, reasons


def create_predictor(student_id: str, type: str = "simplified", 
                 exclude_target: bool = False, exclude_partial: bool = False,
                 tool_set: str = 'full', debug: bool = False):
    return MirrorPredictor(student_id, exclude_target, exclude_partial, tool_set, debug)