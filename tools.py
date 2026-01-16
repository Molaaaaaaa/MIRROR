import os
import json
import re
import shutil
from typing import Dict, List
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import Config
from utils import extract_category


TOOL_SETS = {
    'rag_only': ['rag_search'],
    'rag_stm_ltm': ['rag_search', 'stm_trace', 'ltm_narrative', 'prediction_hint', 
                   'related_questions', 'check_sudden_shift', 'data_quality_info'],
    'full': ['rag_search', 'stm_trace', 'ltm_narrative', 'kg_query',
             'prediction_hint', 'related_questions', 'check_sudden_shift', 
             'data_quality_info', 'category_profile'],
}

_EMBEDDINGS_CACHE = None

def get_embeddings():
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name="SamilPwC-AXNode-GenAI/PwC-Embedding_expr",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _EMBEDDINGS_CACHE


class MirrorToolkit:
    def __init__(self, student_id: str, history: Dict, ltm_data: Dict = None, 
                 kg_data: Dict = None, tool_set: str = 'full'):
        self.student_id = student_id
        self.history = history
        self.ltm_data = ltm_data or {}
        self.kg_data = kg_data or {}
        self.tool_set = tool_set
        self.enabled_tools = TOOL_SETS.get(tool_set, TOOL_SETS['full'])
        
        self._vectorstore = None
        self._vectorstore_loaded = False
        self._category_profiles_cache = None
    
    @property
    def embeddings(self):
        return get_embeddings()
    
    @property
    def vectorstore(self):
        if not self._vectorstore_loaded:
            self._vectorstore = self._load_or_build_vectorstore()
            self._vectorstore_loaded = True
        return self._vectorstore
    
    def _load_or_build_vectorstore(self):
        from langchain_core.documents import Document
        
        persist_dir = os.path.join(Config.CHROMA_DIR, f"mirror_{self.student_id}")
        
        if os.path.exists(persist_dir):
            try:
                return Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=f"mirror_{self.student_id}"
                )
            except Exception:
                shutil.rmtree(persist_dir, ignore_errors=True)
        
        docs = []
        for year_str, year_data in self.history.items():
            year_int = int(year_str)
            if year_int >= Config.TARGET_YEAR:
                continue
            for q, a in year_data.items():
                if len(str(a)) < 100:
                    content = f"[{year_str}년] 질문: {q} | 답변: {a}"
                    docs.append(Document(
                        page_content=content,
                        metadata={"year": year_int, "question": q, "category": extract_category(q)}
                    ))
        
        if not docs:
            return None
        
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name=f"mirror_{self.student_id}"
        )
    
    def _build_category_profiles(self) -> Dict[str, Dict]:
        if self._category_profiles_cache is not None:
            return self._category_profiles_cache
        
        profiles = {}
        patterns = self.ltm_data.get('temporal_patterns', {})
        
        for q, pattern in patterns.items():
            category = pattern.get('topic', extract_category(q))
            
            if category not in profiles:
                profiles[category] = {
                    'last_values': [],
                    'stabilities': [],
                    'questions': []
                }
            
            prof = profiles[category]
            prof['questions'].append(q)
            
            last_val = pattern.get('last_value', '')
            if last_val:
                prof['last_values'].append(last_val)
            
            prof['stabilities'].append(pattern.get('stability', 'unknown'))
        
        for cat, prof in profiles.items():
            if prof['last_values']:
                counter = Counter(prof['last_values'])
                mode_val, mode_count = counter.most_common(1)[0]
                prof['mode'] = mode_val
                prof['mode_ratio'] = mode_count / len(prof['last_values'])
                prof['value_distribution'] = dict(counter)
            else:
                prof['mode'] = None
                prof['mode_ratio'] = 0
                prof['value_distribution'] = {}
            
            stab_scores = {'constant': 1.0, 'highly_stable': 0.8, 'stable': 0.6, 'variable': 0.3}
            scores = [stab_scores.get(s, 0.4) for s in prof['stabilities']]
            prof['avg_stability'] = sum(scores) / len(scores) if scores else 0.5
        
        self._category_profiles_cache = profiles
        return profiles
    
    def _get_related_categories(self, category: str) -> List[str]:
        related = []
        
        corr_path = os.path.join(Config.MIRROR_MEMORY_DIR, "behavioral_correlation.json")
        if os.path.exists(corr_path):
            try:
                with open(corr_path, 'r', encoding='utf-8') as f:
                    corr_data = json.load(f)
                relationships = corr_data.get('category_relationships', {})
                if category in relationships:
                    related = [item[0] for item in relationships[category][:5] 
                               if isinstance(item, (list, tuple)) and len(item) >= 1]
                    if related:
                        return related
            except:
                pass
        
        sim_path = os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
        if os.path.exists(sim_path):
            try:
                with open(sim_path, 'r', encoding='utf-8') as f:
                    sim_data = json.load(f)
                relationships = sim_data.get('category_relationships', {})
                if category in relationships:
                    related = [item[0] for item in relationships[category][:5] 
                               if isinstance(item, (list, tuple)) and len(item) >= 1]
            except:
                pass
        
        return related
    
    def rag_search(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            return "[RAG] 검색 가능한 데이터가 없습니다."
        
        try:
            docs = self.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
            if not docs:
                return "[RAG] 관련 데이터를 찾지 못했습니다."
            return "[RAG 검색결과]\n" + "\n".join([d.page_content for d in docs])
        except:
            return "[RAG] 검색 중 오류가 발생했습니다."
    
    def stm_trace(self, question: str) -> str:
        patterns = self.ltm_data.get('temporal_patterns', {})
        category = extract_category(question)
        
        pattern = patterns.get(question)
        if not pattern:
            q_core = question.split(']')[-1].strip() if ']' in question else question
            q_norm = re.sub(r'\s+', '', q_core.lower())
            for key, p in patterns.items():
                key_core = key.split(']')[-1].strip() if ']' in key else key
                if re.sub(r'\s+', '', key_core.lower()) == q_norm:
                    pattern = p
                    break
        
        if pattern:
            series = pattern.get('series', [])
            if series:
                trace = " -> ".join([f"{s['year']}:{s['value']}" for s in series])
                stability = pattern.get('stability', '')
                last_value = pattern.get('last_value', '')
                mode = pattern.get('mode', '')
                mode_ratio = pattern.get('mode_ratio', 0)
                consecutive = pattern.get('consecutive_same', 0)
                
                result = f"[STM 시계열] {trace}\n"
                result += f"[STM 분석] 안정성:{stability}, 최근값:'{last_value}'"
                
                if mode_ratio >= 0.6:
                    result += f", 주된응답:'{mode}'({mode_ratio:.0%})"
                
                if consecutive >= 3:
                    result += f", 최근{consecutive}년연속동일"
                
                return result
            
        trace_parts = []
        for year in sorted(self.history.keys()):
            if int(year) >= Config.TARGET_YEAR:
                continue
            for q, v in self.history[year].items():
                if question in q or q in question:
                    trace_parts.append(f"{year}:{v}")
                    break
        
        if trace_parts:
            return f"[STM 시계열] {' -> '.join(trace_parts)}"
        
        profiles = self._build_category_profiles()
        related_cats = self._get_related_categories(category)
        
        result_parts = [f"[신규 문항] '{category}' 카테고리의 직접 데이터가 없습니다."]
        
        if related_cats:
            result_parts.append(f"[관련 카테고리] {', '.join(related_cats[:3])}")
            
            for rel_cat in related_cats[:2]:
                if rel_cat in profiles:
                    prof = profiles[rel_cat]
                    mode = prof.get('mode', '')
                    mode_ratio = prof.get('mode_ratio', 0)
                    stab = prof.get('avg_stability', 0.5)
                    
                    stab_desc = "매우안정" if stab >= 0.8 else "안정" if stab >= 0.6 else "변동"
                    
                    if mode and mode_ratio > 0:
                        result_parts.append(f"  - [{rel_cat}] {stab_desc}, 주된응답:'{mode}'({mode_ratio:.0%})")
        
        narrative = self.ltm_data.get('overall_narrative', '')
        if narrative:
            result_parts.append(f"[학생 전반 성향] {narrative[:200]}...")
        
        return "\n".join(result_parts)
    
    def ltm_narrative(self) -> str:
        narrative = self.ltm_data.get('overall_narrative', '')
        if not narrative:
            return "[LTM] 성장 서사가 없습니다."
        
        if isinstance(narrative, str):
            return f"[LTM 전체서사]\n{narrative[:800]}"
        
        return "[LTM] 성장 서사 형식 오류"
    
    def kg_query(self, question: str) -> str:
        if not self.kg_data:
            return "[KG] Knowledge Graph 데이터가 없습니다."
        
        results = []
        
        trends = self.kg_data.get('temporal_trends', {})
        if question in trends:
            trend = trends[question]
            series = trend.get('series', [])[-3:]
            if series:
                results.append(f"KG시계열: {' -> '.join([f'{y}:{v}' for y, v in series])}")
            if trend.get('predicted_next'):
                results.append(f"KG예측값: {trend.get('predicted_next')}")
        
        hints = self.kg_data.get('prediction_hints', {}).get(question, [])
        if hints:
            results.append(f"KG힌트: {hints[0]}")
        
        if not results:
            return "[KG] 해당 질문에 대한 KG 정보가 없습니다."
        
        return "[KG 정보] " + " | ".join(results)
    
    def related_questions(self, question: str, max_count: int = 3) -> str:
        category = extract_category(question)
        patterns = self.ltm_data.get('temporal_patterns', {})
        
        related = []
        for q, data in patterns.items():
            if q == question:
                continue
            if data.get('topic') == category:
                last_value = data.get('last_value', '')
                stability = data.get('stability', '')
                related.append(f"'{last_value}'[{stability}]")
                if len(related) >= max_count:
                    break
        
        if not related:
            related_cats = self._get_related_categories(category)
            for rel_cat in related_cats[:2]:
                for q, data in patterns.items():
                    if data.get('topic') == rel_cat:
                        last_value = data.get('last_value', '')
                        stability = data.get('stability', '')
                        related.append(f"[{rel_cat}]'{last_value}'[{stability}]")
                        if len(related) >= max_count:
                            break
                if len(related) >= max_count:
                    break
        
        if not related:
            return f"[관련문항] '{category}' 카테고리의 관련 문항이 없습니다."

        return f"[관련문항-{category}] 유사 응답 패턴: {', '.join(related)}"
    
    def prediction_hint(self, question: str) -> str:
        hints = self.ltm_data.get('prediction_hints', {})
        
        hint_data = hints.get(question)
        if not hint_data:
            for key in hints.keys():
                if question in key or key in question:
                    hint_data = hints[key]
                    break
        
        if not hint_data:
            return "[예측힌트] 해당 질문에 대한 힌트가 없습니다."
        
        hint_texts = hint_data.get('hints', [])
        confidence = hint_data.get('confidence', '')
        suggested = hint_data.get('suggested_value', '')
        basis = hint_data.get('basis', '')
        
        parts = ["[예측힌트]"]
        if confidence:
            parts.append(f"신뢰도:{confidence}")
        if suggested:
            parts.append(f"추천값:'{suggested}'")
        if hint_texts:
            parts.append(f"근거:{hint_texts[0]}")
        if basis:
            parts.append(f"기반:{basis}")
        
        return " | ".join(parts) if len(parts) > 1 else "[예측힌트] 힌트 없음"
    
    def check_sudden_shift(self, question: str) -> str:
        shifts = self.ltm_data.get('sudden_shifts', {})
        
        shift = shifts.get(question)
        if not shift:
            for key in shifts.keys():
                if question in key or key in question:
                    shift = shifts[key]
                    break
        
        if not shift:
            return "[급변없음] 안정적 패턴 또는 데이터 없음"
        
        year = shift.get('year', '')
        from_val = shift.get('from_value', '')
        to_val = shift.get('to_value', '')
        stability_before = shift.get('stability_before', 0)

        result = f"[급변감지] {year}년: '{from_val}'(이전 {stability_before:.0%}유지) -> '{to_val}'(현재)"
        result += "\n[주의] 2023년 변화지속 또는 이전값 회귀 가능성 고려"
        
        return result
    
    def data_quality_info(self) -> str:
        quality = self.ltm_data.get('data_quality', {})
        if not quality:
            return "[데이터품질] 품질 정보 없음"
        
        difficulty = quality.get('prediction_difficulty', '')
        shift_ratio = quality.get('sudden_shift_ratio', 0)
        stability_avg = quality.get('stability_avg', 0)
        
        result = f"[데이터품질] 예측난이도:{difficulty}, 안정성:{stability_avg:.2f}, 급변율:{shift_ratio:.1%}"
        
        if difficulty == 'easy':
            result += " -> 패턴기반 예측 유효"
        elif difficulty == 'hard':
            result += " -> 급변문항 주의필요"
        
        return result
    
    def category_profile(self, category: str) -> str:
        profiles = self._build_category_profiles()
        
        result_parts = [f"[카테고리 프로파일: {category}]"]
        
        if category in profiles:
            prof = profiles[category]
            mode = prof.get('mode', '')
            mode_ratio = prof.get('mode_ratio', 0)
            stab = prof.get('avg_stability', 0.5)
            dist = prof.get('value_distribution', {})
            
            stab_desc = "매우안정" if stab >= 0.8 else "안정" if stab >= 0.6 else "변동"
            
            result_parts.append(f"안정성: {stab_desc} ({stab:.2f})")
            if mode and mode_ratio > 0:
                result_parts.append(f"주된 응답: '{mode}' ({mode_ratio:.0%})")
            if dist:
                top3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
                result_parts.append(f"응답 분포: {', '.join([f'{v}:{c}건' for v, c in top3])}")
        else:
            result_parts.append(f"'{category}' 카테고리의 직접 데이터가 없습니다.")
        
        related_cats = self._get_related_categories(category)
        if related_cats:
            result_parts.append(f"\n[관련 카테고리]")
            for rel_cat in related_cats[:3]:
                if rel_cat in profiles:
                    prof = profiles[rel_cat]
                    mode = prof.get('mode', '')
                    mode_ratio = prof.get('mode_ratio', 0)
                    stab = prof.get('avg_stability', 0.5)
                    stab_desc = "매우안정" if stab >= 0.8 else "안정" if stab >= 0.6 else "변동"
                    
                    if mode:
                        result_parts.append(f"  - {rel_cat}: {stab_desc}, 주응답 '{mode}'({mode_ratio:.0%})")
        
        return "\n".join(result_parts)
    
    def get_enabled_tool_names(self) -> List[str]:
        return self.enabled_tools.copy()
