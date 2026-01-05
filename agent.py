"""
AI Agent 핵심 모듈 (RAG + AI Agent 기반 - 깊은 추론 버전)
파일명: agent.py

핵심:
1. 관련 카테고리 탐색: RAG/KG 사용 (빠름)
2. 예측: LLM 기반 깊은 추론 (LTM 서사 + 관련 컨텍스트 + 급변 고려)
3. 카테고리별 선택지 유형 인식 (빈도 vs 동의수준)
"""
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter
from langchain_ollama import ChatOllama

from config import Config
from tools import AgentToolkit, TOOL_SETS
from utils import load_student_data, clean_llm_output, get_fallback_prediction, extract_category


# 카테고리 임베딩 유사도 캐시 (신규 카테고리 fallback용)
_CATEGORY_SIMILARITY_CACHE = None

def _load_category_similarity() -> Dict:
    """카테고리 임베딩 유사도 로드 (신규 카테고리 fallback)"""
    global _CATEGORY_SIMILARITY_CACHE
    if _CATEGORY_SIMILARITY_CACHE is None:
        similarity_path = os.path.join(Config.AGENT_MEMORY_DIR, "category_similarity.json")
        if os.path.exists(similarity_path):
            with open(similarity_path, 'r', encoding='utf-8') as f:
                _CATEGORY_SIMILARITY_CACHE = json.load(f)
        else:
            _CATEGORY_SIMILARITY_CACHE = {}
    return _CATEGORY_SIMILARITY_CACHE


# 행동 상관성 캐시 (핵심 - 전체 학생 기반)
_BEHAVIORAL_CORRELATION_CACHE = None

def _load_behavioral_correlation() -> Dict:
    """행동 상관성 데이터 로드 (Behavior-Aware Retrieval)"""
    global _BEHAVIORAL_CORRELATION_CACHE
    if _BEHAVIORAL_CORRELATION_CACHE is None:
        corr_path = os.path.join(Config.AGENT_MEMORY_DIR, "behavioral_correlation.json")
        if os.path.exists(corr_path):
            with open(corr_path, 'r', encoding='utf-8') as f:
                _BEHAVIORAL_CORRELATION_CACHE = json.load(f)
        else:
            _BEHAVIORAL_CORRELATION_CACHE = {}
    return _BEHAVIORAL_CORRELATION_CACHE


# 선택지 유형 판별
FREQUENCY_KEYWORDS = ['없다', '1주일', '한 달', '1년', '번', '회', '빈도']
AGREEMENT_KEYWORDS = ['그렇다', '그렇지 않다', '매우', '전혀', '편이다']


def detect_option_type(options: Dict[str, str]) -> str:
    """선택지 유형 판별 (빈도 vs 동의수준)"""
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
    """선택지 유형에 따른 설명"""
    if option_type == 'frequency':
        return "빈도를 묻는 질문입니다. 경험 횟수/빈도를 선택하세요."
    elif option_type == 'agreement':
        return "동의 수준을 묻는 질문입니다. 해당 정도를 선택하세요."
    else:
        return ""


class SimplifiedAgent:
    """AI Agent (RAG + LLM 깊은 추론)"""
    
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
        
        self.toolkit = AgentToolkit(
            student_id=student_id,
            history=self.history,
            ltm_data=self.ltm_data,
            kg_data=self.kg_data,
            tool_set=tool_set
        )
        
        # LLM 초기화 (깊은 추론을 위해 num_predict 증가)
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=150,
            num_ctx=4096,
            reasoning=False,
            timeout=Config.LLM_TIMEOUT
        )
        
        self._category_profiles = None
        self._available_categories = None
        self._related_categories_cache = {}
        
        # LTM 서사 캐시
        self._narrative_cache = None
        
        if self.debug:
            print(f"[DEBUG] LTM: {len(self.ltm_data.get('temporal_patterns', {}))} patterns")
    
    def _load_data(self, data_type: str) -> Dict:
        if data_type == 'ltm':
            for suffix in ['full_pipeline', 'rich', '']:
                filename = f"{self.student_id}_ltm_{suffix}.json" if suffix else f"{self.student_id}_ltm.json"
                path = os.path.join(Config.AGENT_MEMORY_DIR, filename)
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        else:
            path = os.path.join(Config.AGENT_MEMORY_DIR, f"{self.student_id}_kg.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return {}
    
    def _get_narrative(self) -> str:
        """LTM 서사 조회 (캐시)"""
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
    
    # =========================================================================
    # 관련 카테고리 탐색 (RAG/KG 사용, LLM 불필요)
    # =========================================================================
    
    def _find_related_categories_by_kg(self, category: str) -> Tuple[List[str], Dict[str, float]]:
        """
        행동 상관성 기반 관련 카테고리 조회 (Behavior-Aware)
        
        우선순위:
        1. 전역 행동 상관성 (전체 학생 응답 패턴 기반)
        2. 카테고리 임베딩 유사도 (신규 카테고리 fallback)
        """
        related = []
        scores = {}
        
        # 1순위: 전역 행동 상관성 (핵심)
        behavioral_corr = _load_behavioral_correlation()
        if behavioral_corr:
            relationships = behavioral_corr.get('category_relationships', {})
            if category in relationships:
                for item in relationships[category][:5]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        cat_name, corr = item[0], item[1]
                        related.append(cat_name)
                        scores[cat_name] = abs(corr)  # 절대값 (음의 상관도 중요)
                    elif isinstance(item, (list, tuple)) and len(item) >= 1:
                        related.append(item[0])
                        scores[item[0]] = 0.5
                
                if self.debug and related:
                    print(f"[DEBUG] Behavioral correlation for '{category}': {related[:3]}")
        
        # 2순위: 카테고리 임베딩 유사도 (신규 카테고리 fallback)
        if not related:
            category_sim = _load_category_similarity()
            if category_sim:
                sim_relationships = category_sim.get('category_relationships', {})
                if category in sim_relationships:
                    for item in sim_relationships[category][:5]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            cat_name, score = item[0], item[1]
                            related.append(cat_name)
                            scores[cat_name] = score * 0.8  # 행동 상관성보다 낮은 가중치
                        elif isinstance(item, (list, tuple)) and len(item) >= 1:
                            related.append(item[0])
                            scores[item[0]] = 0.4
                    
                    if self.debug and related:
                        print(f"[DEBUG] Embedding similarity fallback for '{category}': {related[:3]}")
        
        return related, scores
    
    def _find_related_categories_by_similarity(self, category: str) -> Tuple[List[str], Dict[str, float]]:
        """카테고리 임베딩 유사도에서 관련 카테고리 조회 (점수 포함)"""
        related = []
        scores = {}
        
        category_sim = _load_category_similarity()
        if category_sim:
            sim_relationships = category_sim.get('category_relationships', {})
            if category in sim_relationships:
                for item in sim_relationships[category][:5]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        cat_name, score = item[0], item[1]
                        related.append(cat_name)
                        scores[cat_name] = score
                    elif isinstance(item, (list, tuple)) and len(item) >= 1:
                        related.append(item[0])
                        scores[item[0]] = 0.5
                    elif isinstance(item, str):
                        related.append(item)
                        scores[item] = 0.5
        
        return related, scores
    
    def _find_related_categories_by_rag(self, question: str, category: str) -> List[str]:
        """질문 텍스트 기반 RAG 검색"""
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
        """
        Behavior-Aware Retrieval (Voting 전략)
        
        - rag_stm_ltm: RAG만 사용
        - full_agent: 행동 상관성 + RAG 결합
        
        학술적 의의:
        - 텍스트 유사도 "친구가 많다" ↔ "외톨이다" = 유사 (같은 주제)
        - 행동 상관성 "친구가 많다" ↔ "외톨이다" = 음의 상관 (반대 응답)
        """
        if category in self._related_categories_cache:
            return self._related_categories_cache[category]
        
        vote_counter = Counter()
        score_sum = {}
        sources = {}
        
        use_behavioral = 'kg_query' in self.enabled_tools
        
        if use_behavioral:
            # 소스 1: 행동 상관성 (전체 학생 기반, 신규 카테고리는 임베딩 fallback)
            behavioral_cats, behavioral_scores = self._find_related_categories_by_kg(category)
            for cat in behavioral_cats:
                if cat != category:
                    vote_counter[cat] += 1
                    score_sum[cat] = score_sum.get(cat, 0) + behavioral_scores.get(cat, 0.5)
                    sources[cat] = sources.get(cat, []) + ['BEH']
        
        # 소스 2: RAG 검색 (항상 사용)
        rag_cats = self._find_related_categories_by_rag(question, category)
        for cat in rag_cats[:3]:  # RAG는 상위 3개만
            if cat != category:
                vote_counter[cat] += 1
                score_sum[cat] = score_sum.get(cat, 0) + 0.4
                sources[cat] = sources.get(cat, []) + ['RAG']
        
        # Voting + 점수 기반 정렬
        candidates = list(vote_counter.keys())
        candidates.sort(key=lambda x: (vote_counter[x], score_sum.get(x, 0)), reverse=True)
        
        # 결과 선택 (최대 3개로 제한)
        result = candidates[:3]
        
        if self.debug and result:
            debug_info = [f"{cat}({vote_counter[cat]}회,{score_sum.get(cat,0):.2f},{'+'.join(sources.get(cat,[]))})" for cat in result]
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
        
        # 평균 안정성 계산
        for cat, prof in profiles.items():
            if prof['stability_scores']:
                prof['avg_stability'] = sum(prof['stability_scores']) / len(prof['stability_scores'])
            else:
                prof['avg_stability'] = 0.5
        
        self._category_profiles = profiles
        return profiles
    
    def _get_related_context(self, related_cats: List[str]) -> str:
        """관련 카테고리 컨텍스트 생성 (풍부한 정보)"""
        profiles = self._build_category_profiles()
        context_parts = []
        
        for cat in related_cats[:4]:
            if cat in profiles:
                prof = profiles[cat]
                last_vals = prof.get('last_values', [])[:3]
                avg_stab = prof.get('avg_stability', 0)
                
                # 최빈값 계산
                if last_vals:
                    counter = Counter(last_vals)
                    mode_val = counter.most_common(1)[0][0]
                    val_info = f"주요응답:'{mode_val}'"
                else:
                    val_info = ""
                
                stab_desc = "매우안정" if avg_stab >= 0.8 else "안정" if avg_stab >= 0.6 else "변동"
                context_parts.append(f"[{cat}] {stab_desc}, {val_info}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    # =========================================================================
    # LLM 기반 깊은 추론 예측
    # =========================================================================
    
    def _predict_new_question(self, question: str, options: Dict[str, str], category: str) -> Tuple[str, str, Dict]:
        """신규 문항 예측 (LLM 깊은 추론)"""
        ctx = {'is_new_question': True, 'category': category}
        
        # 1. RAG/KG로 관련 카테고리 탐색
        related_cats = self._find_related_categories_dynamic(question, category)
        ctx['related_categories'] = related_cats
        
        # 2. 관련 카테고리 컨텍스트 생성
        related_context = self._get_related_context(related_cats)
        
        # 3. LTM 서사
        narrative = self._get_narrative()
        narrative_text = narrative[:300] if narrative else '정보 없음'
        
        # 4. 옵션 유형 판별 및 문자열 생성
        option_type = detect_option_type(options)
        option_type_desc = get_option_type_description(option_type)
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        # 5. LLM 깊은 추론 프롬프트
        prompt = f"""이 학생의 2018-2022년 종단 데이터를 분석하여 2023년 응답을 예측하세요.

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

[추론 지침]
1. 학생의 전반적인 성향과 발달 궤적을 고려하세요
2. 관련 카테고리의 응답 패턴을 참고하세요
3. 이 학생이 이 질문에 어떻게 응답할지 추론하세요

가장 적절한 번호만 답하세요.
답:"""

        try:
            response = self.llm.invoke(prompt)
            pred = clean_llm_output(response.content)
            
            if pred != "0" and pred in options:
                return pred, f"NewQ:LLM:{','.join(related_cats[:2]) if related_cats else 'none'}", ctx
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM Error: {e}")
        
        # Fallback: 첫 번째 옵션 (보통 '전혀 없다' 또는 '전혀 그렇지 않다')
        option_keys = sorted(options.keys())
        return option_keys[0] if option_keys else "1", "NewQ:Fallback", ctx
    
    def _predict_existing_question(self, question: str, options: Dict[str, str], 
                                    ctx: Dict) -> Tuple[str, str, Dict]:
        """기존 문항 예측 (LLM 깊은 추론)"""
        
        category = extract_category(question)
        stability = ctx.get('stability', '')
        last_value = ctx.get('last_value', '')
        
        # 1. 시계열 히스토리
        hist = self._get_question_history(question)
        if hist:
            trace = " -> ".join([f"{h['year']}:{h['value']}" for h in hist])
        else:
            trace = f"최근값: {last_value}" if last_value else "정보 없음"
        
        # 2. 급변 여부 확인
        shift = self._get_sudden_shift(question)
        shift_warning = ""
        if shift:
            shift_year = shift.get('year', '')
            from_val = shift.get('from_value', '')
            to_val = shift.get('to_value', '')
            shift_warning = f"\n[주의] {shift_year}년 급변 감지: '{from_val}' -> '{to_val}' (변화 지속 또는 회귀 가능성 고려)"
        
        # 3. LTM 서사 (짧게)
        narrative = self._get_narrative()
        narrative_short = narrative[:200] if narrative else ''
        
        # 4. 관련 카테고리 컨텍스트
        related_cats = self._find_related_categories_dynamic(question, category)
        related_context = self._get_related_context(related_cats[:3])
        
        # 5. 옵션 유형 판별 및 문자열 생성
        option_type = detect_option_type(options)
        option_type_desc = get_option_type_description(option_type)
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        # 6. LLM 깊은 추론 프롬프트
        prompt = f"""이 학생의 시계열 데이터를 분석하여 2023년 응답을 예측하세요.

[학생 개요]
{narrative_short if narrative_short else '정보 없음'}

[이 문항의 시계열 (2018-2022)]
{trace}
패턴 안정성: {stability}
{shift_warning}

[관련 카테고리]
{related_context if related_context else ''}

[예측 대상]
질문: {question}

[선택지]
{options_str}
({option_type_desc})

[추론 지침]
1. 시계열 패턴을 분석하되, 단순히 마지막 값을 반복하지 마세요
2. 급변이 감지되었다면 변화 지속 vs 이전 패턴 회귀를 신중히 판단하세요
3. 학생의 전반적 발달 궤적과 일관성을 고려하세요

가장 적절한 번호만 답하세요.
답:"""

        try:
            response = self.llm.invoke(prompt)
            pred = clean_llm_output(response.content)
            
            if pred != "0" and pred in options:
                method = f"LLM:{stability}"
                if shift:
                    method += ":shift"
                return pred, method, ctx
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM Error: {e}")
        
        # Fallback
        if ctx.get('suggested_value'):
            pred_key = self._find_option_key(ctx['suggested_value'], options)
            if pred_key != "0":
                return pred_key, "Fallback:Suggested", ctx
        
        if last_value:
            pred_key = self._find_option_key(last_value, options)
            if pred_key != "0":
                return pred_key, "Fallback:LastValue", ctx
        
        hist = self._get_question_history(question)
        pred = get_fallback_prediction(hist, options)
        return pred, "Fallback:Final", ctx
    
    def _build_structured_context(self, question: str) -> Dict:
        """구조화된 컨텍스트 생성"""
        context = {
            'stability': None,
            'last_value': None,
            'suggested_value': None,
            'confidence': None,
            'has_sudden_shift': False,
            'is_new_question': False
        }
        
        # LTM 패턴 조회
        pattern = self._get_temporal_pattern(question)
        if pattern:
            context['stability'] = pattern.get('stability', 'unknown')
            context['last_value'] = pattern.get('last_value', '')
            context['has_sudden_shift'] = pattern.get('has_sudden_shift', False)
        
        # 예측 힌트 조회
        hint = self._get_prediction_hint(question)
        if hint:
            context['suggested_value'] = hint.get('suggested_value', '')
            context['confidence'] = hint.get('confidence', 'medium')
        
        # 급변 조회
        shift = self._get_sudden_shift(question)
        if shift:
            context['has_sudden_shift'] = True
        
        # LTM에 없으면 히스토리에서 직접 분석
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
        """예측 실행 (LLM 깊은 추론)"""
        category = extract_category(question)
        ctx = self._build_structured_context(question)
        
        if self.debug:
            print(f"\n[DEBUG] Q: {question[:50]}...")
            print(f"[DEBUG] Cat: {category}, Stability: {ctx.get('stability')}, Shift: {ctx.get('has_sudden_shift')}")
            print(f"[DEBUG] Options: {options}")
        
        # 신규 문항: LLM 깊은 추론
        if ctx['is_new_question']:
            return self._predict_new_question(question, options, category)
        
        # 기존 문항: LLM 깊은 추론
        return self._predict_existing_question(question, options, ctx)
    
    def predict_batch(self, tasks: List[Dict], verbose: bool = False) -> Tuple[List[str], List[str]]:
        from tqdm import tqdm
        preds, reasons = [], []
        
        for task in tqdm(tasks, desc=f"[{self.tool_set}] {self.student_id}"):
            pred, reason, _ = self.predict(task['question'], task['options'])
            preds.append(pred)
            reasons.append(reason)
        
        return preds, reasons


def create_agent(student_id: str, agent_type: str = "simplified", 
                 exclude_target: bool = False, exclude_partial: bool = False,
                 tool_set: str = 'full', debug: bool = False):
    return SimplifiedAgent(student_id, exclude_target, exclude_partial, tool_set, debug)
