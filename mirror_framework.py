"""
MIRROR Framework Core Module

Architecture:
- Offline Stage:
  1. RER (Retrospective Evidence Retrieval): Q-A-Year document cache
  2. LTE (Longitudinal Trend Extraction): Static + Dynamic Persona
  3. KG (Knowledge Graph): Domain-Item-Option hierarchy + Correlation edges

- Online Stage:
  4. Consistency-aware Prediction: Dynamic context construction + Single-step inference
     CTX_{u,q} = (q, Y_q, E_{u,q}, P_{u,q}, C_q)

- Each component (RER/LTE/KG) is strictly isolated
- KG component provides ONLY correlation constraints (C_q), NOT temporal data
- LTE component provides temporal patterns (P^year), stability, and persona
- RER filters out self-referential documents
- Fallback strategies respect component flags
"""
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from data_constants import (
    FREQUENCY_KEYWORDS,
    AGREEMENT_KEYWORDS,
    RELATED_CATEGORY_PRIORITY,
)

from config import Config
from tools import MirrorToolkit, TOOL_SETS
from utils import load_student_data, clean_llm_output, get_fallback_prediction, extract_category


def get_llm(temperature: float = 0.0, max_tokens: int = 250, timeout: int = None):
    """
    Return appropriate LLM instance based on Config.LLM_PROVIDER.

    Supported providers:
    - ollama: Local Ollama (Gemma3, Qwen3, etc.)
    - openai: OpenAI API (GPT-4o-mini, etc.)
    - deepseek: DeepSeek API (deepseek-chat / DeepSeek-V3)
    """
    provider = getattr(Config, 'LLM_PROVIDER', 'ollama')
    timeout = timeout or getattr(Config, 'LLM_TIMEOUT', 600)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

    else:  # ollama (default)
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=getattr(Config, 'NUM_CTX', 4096),
            reasoning=False,
            timeout=timeout
        )

# Import KG class for actual graph queries
try:
    from build_kg import MirrorKnowledgeGraph
    HAS_KG_CLASS = True
except ImportError:
    HAS_KG_CLASS = False
    MirrorKnowledgeGraph = None


# Global caches
_CATEGORY_SIMILARITY_CACHE = None
_BEHAVIORAL_CORRELATION_CACHE = None
_KNOWLEDGE_GRAPH_CACHE = {}  # student_id -> MirrorKnowledgeGraph


def _manage_kg_cache_size():
    """Manage KG cache size to prevent memory leaks."""
    global _KNOWLEDGE_GRAPH_CACHE
    max_size = getattr(Config, 'KG_CACHE_MAX_SIZE', 50)
    if len(_KNOWLEDGE_GRAPH_CACHE) > max_size:
        # Remove oldest half of entries
        keys_to_remove = list(_KNOWLEDGE_GRAPH_CACHE.keys())[:max_size // 2]
        for k in keys_to_remove:
            del _KNOWLEDGE_GRAPH_CACHE[k]


def _load_category_similarity() -> Dict:
    global _CATEGORY_SIMILARITY_CACHE
    if _CATEGORY_SIMILARITY_CACHE is None:
        path = os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                _CATEGORY_SIMILARITY_CACHE = json.load(f)
        else:
            _CATEGORY_SIMILARITY_CACHE = {}
    return _CATEGORY_SIMILARITY_CACHE


def _load_behavioral_correlation() -> Dict:
    global _BEHAVIORAL_CORRELATION_CACHE
    if _BEHAVIORAL_CORRELATION_CACHE is None:
        path = os.path.join(Config.MIRROR_MEMORY_DIR, "behavioral_correlation.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                _BEHAVIORAL_CORRELATION_CACHE = json.load(f)
        else:
            _BEHAVIORAL_CORRELATION_CACHE = {}
    return _BEHAVIORAL_CORRELATION_CACHE


def _load_knowledge_graph(student_id: str, kg_data: Dict) -> Optional['MirrorKnowledgeGraph']:
    """Load or reconstruct MirrorKnowledgeGraph for actual graph queries."""
    global _KNOWLEDGE_GRAPH_CACHE
    
    if not HAS_KG_CLASS or MirrorKnowledgeGraph is None:
        return None
    
    if student_id in _KNOWLEDGE_GRAPH_CACHE:
        return _KNOWLEDGE_GRAPH_CACHE[student_id]
    
    # Manage cache size before adding new entry
    _manage_kg_cache_size()
    
    # Try to reconstruct from kg_data['knowledge_graph']
    graph_data = kg_data.get('knowledge_graph', {})
    if graph_data:
        try:
            kg = MirrorKnowledgeGraph.from_dict(graph_data)
            _KNOWLEDGE_GRAPH_CACHE[student_id] = kg
            return kg
        except Exception:
            pass
    
    return None


def detect_option_type(options: Dict[str, str]) -> str:
    option_text = ' '.join(options.values())
    freq_score = sum(1 for kw in FREQUENCY_KEYWORDS if kw in option_text)
    agree_score = sum(1 for kw in AGREEMENT_KEYWORDS if kw in option_text)
    
    if freq_score > agree_score:
        return 'frequency'
    elif agree_score > 0:
        return 'agreement'
    return 'other'


def get_option_type_description(option_type: str) -> str:
    if option_type == 'frequency':
        return "발생 빈도를 선택하세요."
    elif option_type == 'agreement':
        return "동의 수준을 선택하세요."
    return ""


class MirrorPredictor:
    """
    Components:
    - use_rer: Retrospective Evidence Retrieval
    - use_lte: Longitudinal Trend Extraction
    - use_kg: Knowledge Graph Constraints
    
    IMPORTANT: Components are strictly isolated for valid ablation study.
    - KG provides ONLY correlation constraints (C_q), NOT temporal patterns
    - LTE provides temporal patterns (P^year), stability, and persona
    - RER filters out self-referential documents
    - Fallback strategies respect component flags
    """
    
    def __init__(self, student_id: str,
                 exclude_target: bool = False,
                 exclude_partial: bool = False,
                 use_rer: bool = True,
                 use_lte: bool = True,
                 use_kg: bool = True,
                 tool_set: str = 'full',
                 debug: bool = False):

        self.student_id = student_id
        self.debug = debug

        # Component flags
        self.use_rer = use_rer
        self.use_lte = use_lte
        self.use_kg = use_kg
        self.tool_set = tool_set
        self.enabled_tools = TOOL_SETS.get(tool_set, TOOL_SETS['full'])

        # Determine experimental setting for cache isolation
        if exclude_partial:
            self._setting = 'S3'
        elif exclude_target:
            self._setting = 'S2'
        else:
            self._setting = 'S1'

        # Load student data
        _, self.history = load_student_data(
            Config.DATA_DIR, student_id,
            exclude_target=exclude_target,
            exclude_partial=exclude_partial
        )

        # Load LTE data (P_u: Persona profiles) - ONLY if use_lte=True
        self.ltm_data = self._load_data('ltm') if use_lte else {}

        # Load KG data - ONLY if use_kg=True
        self.kg_data = self._load_data('kg') if use_kg else {}

        # Initialize toolkit for RER (evidence retrieval) - ONLY if use_rer=True
        self.toolkit = MirrorToolkit(
            student_id=student_id,
            history=self.history,
            ltm_data=self.ltm_data if use_lte else {},
            kg_data=self.kg_data if use_kg else {},
            tool_set=tool_set,
            setting=self._setting
        ) if use_rer else None
        
        # LLM for inference (supports Ollama, OpenAI, DeepSeek)
        self.llm = get_llm(temperature=0.0, max_tokens=20)
        
        # Caches
        self._category_profiles = None
        self._related_categories_cache = {}
        self._narrative_cache = None
        self._embedding_cache = {}  # question -> embedding vector
        self._patterns_by_topic = self._build_topic_index()

        if self.debug:
            print(f"[MIRROR] Student: {student_id}")
            print(f"[MIRROR] Components: RER={use_rer}, LTE={use_lte}, KG={use_kg}")
            print(f"[MIRROR] History years: {list(self.history.keys())}")
            if use_lte:
                print(f"[MIRROR] LTM patterns: {len(self.ltm_data.get('temporal_patterns', {}))}")
            if use_kg:
                print(f"[MIRROR] KG trends: {len(self.kg_data.get('temporal_trends', {}))}")
    
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
    
    def _build_topic_index(self) -> Dict[str, list]:
        """Pre-index temporal_patterns by topic for O(1) lookup."""
        index = {}
        for q, pattern in self.ltm_data.get('temporal_patterns', {}).items():
            topic = pattern.get('topic', '')
            if topic:
                if topic not in index:
                    index[topic] = []
                index[topic].append((q, pattern))
        return index

    # Retrospective Evidence Retrieval (RER)
    def precompute_embeddings(self, questions: list):
        """Pre-compute embeddings for all target questions at once."""
        if not self.use_rer or not self.toolkit or not self.toolkit.vectorstore:
            return
        uncached = [q for q in questions if q not in self._embedding_cache]
        if uncached:
            embeddings = self.toolkit.embeddings.embed_documents(uncached)
            for q, emb in zip(uncached, embeddings):
                self._embedding_cache[q] = emb

    def _retrieve_evidence(self, question: str, k: int = 5) -> str:
        """
        RER: Retrieve top-K relevant Q-A-Year documents for the target question.
             E_{u,q} = R(z_q, D_u, K)
        Paper Section 2.1: semantic search over Q-A-Year document cache.
        """
        if not self.use_rer or not self.toolkit or not self.toolkit.vectorstore:
            return ""

        try:
            # Use cached embedding if available, otherwise compute
            if question in self._embedding_cache:
                docs = self.toolkit.vectorstore.similarity_search_by_vector(
                    self._embedding_cache[question], k=k + 3)
            else:
                docs = self.toolkit.vectorstore.as_retriever(
                    search_kwargs={"k": k + 3}
                ).invoke(question)

            if not docs:
                return ""

            evidence_parts = []
            target_q_norm = self._normalize_for_match(question)
            target_q_core = question.split(']')[-1].strip() if ']' in question else question
            target_q_core_norm = self._normalize_for_match(target_q_core)

            for doc in docs:
                doc_question = doc.metadata.get('question', '')
                doc_q_norm = self._normalize_for_match(doc_question)
                doc_q_core = doc_question.split(']')[-1].strip() if ']' in doc_question else doc_question
                doc_q_core_norm = self._normalize_for_match(doc_q_core)

                if doc_q_norm == target_q_norm or doc_q_core_norm == target_q_core_norm:
                    continue

                evidence_parts.append(doc.page_content)
                if len(evidence_parts) >= k:
                    break

            return "\n".join(evidence_parts)
        except Exception as e:
            if self.debug:
                print(f"[RER] Error: {e}")
            return ""

    def _count_stable_evidence(self, evidence_parts: list) -> int:
        """Count how many evidence items come from stable patterns."""
        patterns = self.ltm_data.get('temporal_patterns', {})
        if not patterns:
            return 0
        count = 0
        for part in evidence_parts:
            if '질문:' in part:
                q_part = part.split('질문:')[1].split('|')[0].strip() if '|' in part else part.split('질문:')[1].strip()
                for q, pattern in patterns.items():
                    if q_part in q or q in q_part:
                        stab = pattern.get('stability', '')
                        if stab in ('constant', 'highly_stable', 'stable'):
                            count += 1
                        break
        return count

    def _summarize_cross_category_with_stability(self, cross_cat_docs: list,
                                                   target_category: str) -> str:
        """
        LTE-enhanced cross-category summary for cold-start.
        Adds stability information to make evidence more reliable.
        """
        from collections import Counter
        patterns = self.ltm_data.get('temporal_patterns', {})

        low_keywords = ['전혀', '없다', '않는다', '않았다', '한 번도']
        high_keywords = ['그런 편이다', '자주', '매우', '많이', '그렇다']

        cat_info = {}  # cat -> {'levels': [], 'stable': bool}
        for doc_info in cross_cat_docs:
            content = doc_info['content']
            cat = doc_info['category']
            if '답변:' in content:
                answer = content.split('답변:')[-1].strip()
                if any(kw in answer for kw in low_keywords):
                    level = '낮음'
                elif any(kw in answer for kw in high_keywords):
                    level = '높음'
                else:
                    level = '중간'
                if cat not in cat_info:
                    cat_info[cat] = {'levels': [], 'stable': False}
                cat_info[cat]['levels'].append(level)

        # Check stability from LTE patterns
        if patterns:
            for cat in cat_info:
                stable_count = 0
                total = 0
                for q, pattern in patterns.items():
                    if pattern.get('topic') == cat:
                        total += 1
                        stab = pattern.get('stability', '')
                        if stab in ('constant', 'highly_stable', 'stable'):
                            stable_count += 1
                if total > 0 and stable_count / total >= 0.5:
                    cat_info[cat]['stable'] = True

        if not cat_info:
            return ""

        parts = ["[관련 카테고리 행동 경향 요약]"]
        for cat, info in cat_info.items():
            level_counts = Counter(info['levels'])
            dominant = level_counts.most_common(1)[0][0]
            total = len(info['levels'])
            stab_mark = " (안정적)" if info['stable'] else ""
            parts.append(f"- {cat}: {total}건 기록 중 '{dominant}' 수준{stab_mark}")
        parts.append("(주의: 다른 카테고리 경향이므로 현재 문항과 직접 비교 불가)")
        return "\n".join(parts)

    def _summarize_cross_category_with_direction(self, cross_cat_docs: list,
                                                    target_category: str) -> str:
        """
        RER cold-start: Summarize cross-category evidence with scale direction.

        Key problem: categories have different scale directions.
        - Positive scales (친구관계, 교사관계, 삶의만족도): high = good
        - Negative scales (공격성, 폭력, 비행): high = bad
        Without direction info, the model sees "친구관계: 높음" and wrongly
        infers "폭력도 높음". With direction, it can reason correctly:
        "친구관계(긍정) 높음 → 부정적 행동은 반대 방향 → 낮음"
        """
        # Categorize scales by direction
        negative_cats = {'공격성', '우울', '학업 무기력', '사회적 위축', '주의집중',
                         '신체증상', '비일관성', '현실비행 경험 유무 및 빈도',
                         '사이버비행 경험 유무 및 빈도', '거부', '학교 폭력'}
        positive_cats = {'친구관계', '교사관계', '삶의 만족도', '자아존중감', '자아탄력성',
                         '그릿(Grit)', '학업 열의', '협동심', '구조제공', '애정'}

        low_keywords = ['전혀', '없다', '않는다', '않다', '하지 않', '않은 편', '거의 없']
        high_keywords = ['그런 편이다', '자주', '매우', '많이', '그렇다']

        from collections import Counter
        cat_info = {}
        for doc_info in cross_cat_docs:
            content = doc_info['content']
            cat = doc_info['category']
            if '답변:' in content:
                answer = content.split('답변:')[-1].strip()
                if any(kw in answer for kw in low_keywords):
                    level = '낮음'
                elif any(kw in answer for kw in high_keywords):
                    level = '높음'
                else:
                    level = '중간'
                if cat not in cat_info:
                    cat_info[cat] = []
                cat_info[cat].append(level)

        if not cat_info:
            return ""

        # Determine target question's scale direction
        target_is_negative = target_category in negative_cats

        parts = ["[다른 카테고리 응답 기록]"]
        for cat, levels in cat_info.items():
            dominant = Counter(levels).most_common(1)[0][0]
            total = len(levels)
            # Annotate scale direction
            if cat in positive_cats:
                direction = "긍정 척도"
            elif cat in negative_cats:
                direction = "부정 척도"
            else:
                direction = ""
            dir_str = f" ({direction})" if direction else ""
            parts.append(f"- {cat}{dir_str}: {total}건 기록 중 '{dominant}' 수준")

        return "\n".join(parts)

    def _rerank_by_stability(self, evidence_parts: list, k: int) -> list:
        """
        LTE: Rerank evidence by preferring items from stable patterns.
        Stable patterns provide more reliable reference data.
        """
        patterns = self.ltm_data.get('temporal_patterns', {})
        if not patterns:
            return evidence_parts

        scored = []
        for part in evidence_parts:
            # Extract question from evidence format: "[year년] 질문: Q | 답변: A"
            score = 0
            if '질문:' in part:
                q_part = part.split('질문:')[1].split('|')[0].strip() if '|' in part else part.split('질문:')[1].strip()
                for q, pattern in patterns.items():
                    if q_part in q or q in q_part:
                        stab = pattern.get('stability', 'unknown')
                        if stab in ('constant', 'highly_stable'):
                            score = 2
                        elif stab == 'stable':
                            score = 1
                        break
            scored.append((score, part))

        scored.sort(key=lambda x: -x[0])
        return [part for _, part in scored[:k]]

    def _summarize_cross_category_evidence(self, cross_cat_docs: list, target_category: str) -> str:
        """
        Summarize cross-category evidence as behavioral tendency instead of raw answers.
        This prevents Gemma3 from being misled by different-scale answers.

        Instead of: "[2022년] 질문: 공격성-화를 잘 낸다 | 답변: 그런 편이다"
        Returns:   "관련 카테고리(공격성) 기록에서 이 학생은 대체로 낮은~중간 수준의 행동 경향을 보입니다."
        """
        from collections import Counter

        # Parse answer levels from cross-category docs
        cat_answers = {}  # category -> list of (question_summary, answer_level)

        low_keywords = ['전혀', '없다', '않는다', '않았다', '한 번도']
        mid_keywords = ['그렇지 않은', '가끔', '별로']
        high_keywords = ['그런 편이다', '자주', '매우', '많이', '그렇다']

        for doc_info in cross_cat_docs:
            content = doc_info['content']
            cat = doc_info['category']

            # Extract answer from content format: "[year년] 질문: ... | 답변: ..."
            if '답변:' in content:
                answer = content.split('답변:')[-1].strip()
                if any(kw in answer for kw in low_keywords):
                    level = '낮음'
                elif any(kw in answer for kw in high_keywords):
                    level = '높음'
                else:
                    level = '중간'

                if cat not in cat_answers:
                    cat_answers[cat] = []
                cat_answers[cat].append(level)

        if not cat_answers:
            return ""

        # Build summary
        parts = ["[관련 카테고리 행동 경향 요약]"]
        for cat, levels in cat_answers.items():
            level_counts = Counter(levels)
            dominant = level_counts.most_common(1)[0][0]
            total = len(levels)
            parts.append(f"- {cat}: {total}건 기록 중 대체로 '{dominant}' 수준")

        parts.append("(주의: 위는 다른 카테고리의 경향이므로 현재 문항의 척도와 직접 비교하지 마세요)")
        return "\n".join(parts)

    def _get_simple_related_summary(self, target_category: str) -> str:
        """
        RER cold-start: Get simple summary of related category responses.
        No KG needed — uses RELATED_CATEGORY_PRIORITY and raw history.

        Returns: concise behavior level summary from related categories.
        """
        if not RELATED_CATEGORY_PRIORITY:
            return ""

        # Check multiple related categories for a fuller picture
        all_values = []
        cat_used = None
        for rel_cat in RELATED_CATEGORY_PRIORITY[:3]:
            last_values = []
            for year in sorted(self.history.keys(), reverse=True):
                if int(year) >= Config.TARGET_YEAR:
                    continue
                for q, v in self.history[year].items():
                    q_cat = extract_category(q)
                    if q_cat == rel_cat and v:
                        last_values.append(v)
                if last_values:
                    break
            if last_values:
                all_values.extend(last_values)
                if cat_used is None:
                    cat_used = rel_cat

        if not all_values:
            return ""

        # Count low-level responses across all related categories
        low_keywords = ['전혀', '없다', '않는다', '않다', '하지 않', '않은 편', '거의 없', '거의 하지']
        low_count = sum(1 for v in all_values if any(kw in v for kw in low_keywords))
        low_ratio = low_count / len(all_values)

        if low_ratio >= 0.7:
            return f"[참고] 이 학생은 관련 행동({cat_used} 등) 항목에서 대부분 낮은 수준으로 응답함"
        elif low_ratio >= 0.4:
            return f"[참고] 이 학생은 관련 행동 항목에서 낮은~중간 수준으로 응답함"
        else:
            mode_val = Counter(all_values).most_common(1)[0]
            return f"[참고] 이 학생의 관련 행동({cat_used}) 최빈 응답: '{mode_val[0]}'"

    def _summarize_cross_category_concise(self, cross_cat_docs: list) -> str:
        """
        RER cold-start: Show representative Q-A pairs per category.
        Uses original answer text (not abstracted levels) to avoid
        misleading the LLM with wrong high/low classification.
        Limits to 1 representative per category, max 3 categories.
        """
        cat_examples = {}  # cat -> first (question_short, answer) pair
        for doc_info in cross_cat_docs:
            content = doc_info['content']
            cat = doc_info['category']
            if cat in cat_examples:
                continue  # one per category
            if '질문:' in content and '답변:' in content:
                q_part = content.split('질문:')[1].split('|')[0].strip()
                # Remove category prefix from question
                if ']-' in q_part:
                    q_part = q_part.split(']-')[1].strip()
                elif '-' in q_part:
                    q_part = q_part.split('-', 1)[1].strip() if q_part.startswith('[') else q_part
                answer = content.split('답변:')[-1].strip()
                cat_examples[cat] = (q_part[:40], answer)
            if len(cat_examples) >= 3:
                break

        if not cat_examples:
            return ""

        parts = []
        for cat, (q_short, ans) in cat_examples.items():
            parts.append(f"- [{cat}] {q_short} → {ans}")
        return "\n".join(parts)

    def _get_concise_lte_tendency(self, ctx: Dict) -> str:
        """
        Ultra-concise LTE contribution for cold-start: ONE line about student tendency.
        Uses narrative + domain profiles to determine if student shows
        high/low risk behavior pattern overall.
        """
        if not self.use_lte:
            return ""

        # Check domain profiles for negative behavior domains
        profiles = self.ltm_data.get('thematic_profiles', {})
        neg_domains = ['공격성', '현실비행 경험 유무 및 빈도']
        risk_level = "low"

        for domain in neg_domains:
            if domain in profiles:
                prof = profiles[domain]
                stab_score = prof.get('stability_score', 0.5)
                # Check if the student shows high levels in negative domains
                level = prof.get('overall_level', '')
                if level in ['high', '높음']:
                    risk_level = "high"
                    break
                elif level in ['mid', '중간']:
                    risk_level = "mid"

        # Also check from temporal patterns
        if risk_level == "low":
            patterns = self.ltm_data.get('temporal_patterns', {})
            high_count = 0
            total_neg = 0
            for q, pattern in patterns.items():
                topic = pattern.get('topic', '')
                if topic in neg_domains:
                    total_neg += 1
                    level = self._get_response_level_from_pattern(pattern)
                    if level == "high":
                        high_count += 1
            if total_neg > 0 and high_count / total_neg > 0.3:
                risk_level = "mid"

        if risk_level == "high":
            return "[LTE] 이 학생의 전반적 부정적 행동 경향: 높음"
        elif risk_level == "mid":
            return "[LTE] 이 학생의 전반적 부정적 행동 경향: 중간"
        else:
            return "[LTE] 이 학생의 전반적 부정적 행동 경향: 낮음"

    # Longitudinal Trend Extraction (LTE)

    def _get_static_persona(self) -> str:
        """P^static_u: Demographics and stable traits"""
        if not self.use_lte:
            return ""
        
        static = self.ltm_data.get('static_persona', {})
        if not static or static == {'raw': {}}:
            return ""
        
        parts = []
        if static.get('gender'):
            parts.append(f"성별: {static['gender']}")
        if static.get('birth_year'):
            parts.append(f"출생년도: {static['birth_year']}")
        if static.get('region'):
            parts.append(f"지역: {static['region']}")
        if static.get('school_region'):
            parts.append(f"학교지역: {static['school_region']}")
        if static.get('city_size'):
            parts.append(f"도시규모: {static['city_size']}")
        if static.get('siblings'):
            parts.append(f"형제자매수: {static['siblings']}")

        return ", ".join(parts) if parts else ""
    
    def _get_domain_profile(self, category: str) -> str:
        """P^domain_u: Domain-level trend summary"""
        if not self.use_lte:
            return ""
        
        profiles = self.ltm_data.get('thematic_profiles', {})
        
        prof = profiles.get(category)
        
        # Try partial match for Korean category names
        if not prof:
            for key in profiles:
                if category in key or key in category:
                    prof = profiles[key]
                    break
        
        if not prof:
            return ""
        
        stability = prof.get('stability_score', 0)
        trend = prof.get('dominant_trend', 'unknown')
        shift_ratio = prof.get('sudden_shift_ratio', 0)
        q_count = prof.get('question_count', 0)
        
        stab_desc = "매우안정" if stability >= 0.8 else "안정" if stability >= 0.6 else "변동"
        
        parts = [f"[{category} 도메인 프로필]"]
        parts.append(f"- 전체 안정성: {stab_desc} ({stability:.2f})")
        parts.append(f"- 지배적 추세: {trend}")
        if shift_ratio > 0:
            parts.append(f"- 급변 비율: {shift_ratio:.0%} ({q_count}개 문항 중)")
        
        return "\n".join(parts)
    
    def _get_yearly_profile(self, question: str) -> Tuple[str, Optional[Dict]]:
        """P^year_u: Item-level temporal trace and stability"""
        # P^year is LTE component - strict isolation for ablation
        if not self.use_lte:
            return "", None
        
        # With LTE, try to get enriched pattern data from ltm_data
        pattern = self._find_in_dict(question, self.ltm_data.get('temporal_patterns', {}))
        
        if pattern:
            series = pattern.get('series', [])
            if series:
                trace = " -> ".join([f"{s['year']}:{s['value']}" for s in series])
            else:
                trace = f"최근값: {pattern.get('last_value', '')}"
            return trace, pattern
        
        # Fallback to raw history (still within LTE scope)
        hist = self._get_question_history(question)
        if hist:
            trace = " -> ".join([f"{h['year']}:{h['value']}" for h in hist])
            return trace, None
        
        return "", None
    
    def _get_change_profile(self, question: str) -> str:
        """P^change_u: Sudden shift detection and warning"""
        if not self.use_lte:
            return ""
        
        shift = self._find_in_dict(question, self.ltm_data.get('sudden_shifts', {}))
        if shift:
            year = shift.get('year', '')
            from_val = shift.get('from_value', '')
            to_val = shift.get('to_value', '')
            return f"[주의] {year}년 급변: '{from_val}' -> '{to_val}'"

        pattern = self._find_in_dict(question, self.ltm_data.get('temporal_patterns', {}))
        if pattern and pattern.get('has_sudden_shift', False):
            series = pattern.get('series', [])
            if len(series) >= 2:
                prev_val = series[-2].get('value', '')
                curr_val = series[-1].get('value', '')
                curr_year = series[-1].get('year', '')
                if prev_val != curr_val:
                    return f"[주의] {curr_year}년 급변: '{prev_val}' -> '{curr_val}'"

        return ""
    
    def _get_overall_narrative(self) -> str:
        """Overall growth narrative from LTE"""
        if not self.use_lte:
            return ""
        
        if self._narrative_cache is None:
            self._narrative_cache = self.ltm_data.get('overall_narrative', '')
        return self._narrative_cache
    
    # Knowledge Graph Constraints (KG)
    
    def _get_kg_constraints(self, question: str, category: str) -> str:
        """
        C_q: KG-derived consistency constraints.
        - Behavioral correlations (Pearson)
        - Semantic similarity edges
        """
        if not self.use_kg:
            return ""
        
        constraints = []
        
        # Student-specific KG correlations
        kg_relationships = self.kg_data.get('category_relationships', {})
        if category in kg_relationships:
            for item in kg_relationships[category][:3]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    rel_cat, corr = item[0], item[1]
                    direction = "양의" if corr > 0 else "음의"
                    strength = "강한" if abs(corr) >= 0.5 else "중간" if abs(corr) >= 0.3 else "약한"
                    constraints.append(f"- {category} ↔ {rel_cat}: {strength} {direction} 상관 ({corr:.2f})")
        
        # Global behavioral correlation (fallback)
        if not constraints:
            behavioral_corr = _load_behavioral_correlation()
            if behavioral_corr:
                relationships = behavioral_corr.get('category_relationships', {})
                if category in relationships:
                    for item in relationships[category][:3]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            rel_cat, corr = item[0], item[1]
                            direction = "양의" if corr > 0 else "음의"
                            strength = "강한" if abs(corr) >= 0.5 else "중간" if abs(corr) >= 0.3 else "약한"
                            constraints.append(f"- {category} ↔ {rel_cat}: {strength} {direction} 상관 ({corr:.2f})")
        
        # Semantic similarity (fallback)
        if not constraints:
            category_sim = _load_category_similarity()
            if category_sim:
                sim_rels = category_sim.get('category_relationships', {}).get(category, [])
                for item in sim_rels[:3]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rel_cat, sim = item[0], item[1]
                        constraints.append(f"- {category} ~ {rel_cat}: 의미적 유사도 ({sim:.2f})")
        
        if constraints:
            return "[KG 제약조건]\n" + "\n".join(constraints)
        return ""

    def _get_kg_valid_option_range(self, question: str) -> str:
        """
        KG-derived valid option range constraint (Y_q from KG schema).
        Paper Section 2.4: KG explicitly provides valid response range.
        """
        if not self.use_kg:
            return ""

        kg = _load_knowledge_graph(self.student_id, self.kg_data)
        if kg is None:
            return ""

        try:
            valid_options = kg.get_question_options(question)
            if valid_options:
                option_range = ", ".join([f"{k}: {v}" for k, v in sorted(valid_options.items())])
                return f"[KG 유효 응답 범위] {option_range}"
        except Exception:
            pass
        return ""

    def _get_related_category_context(self, question: str, category: str) -> str:
        """
        KG's role is to provide CORRELATION information between categories,
        NOT temporal patterns or stability (that's LTE's role in P^year).
        
        This ensures:
        1. Clean separation: KG = correlations, LTE = temporal patterns
        2. Valid ablation: MIRROR - RER_LTE = pure KG contribution (correlations only)
        3. No noise from incomplete temporal data in kg_data
        """
        if not self.use_kg:
            return ""
        
        related_cats = self._find_related_categories(question, category)
        if not related_cats:
            return ""
        
        # KG provides ONLY correlation information
        # NOT temporal patterns - that's LTE's responsibility
        context_parts = []
        
        for rel_cat in related_cats[:3]:
            corr_info = self._get_correlation_info(category, rel_cat)
            if corr_info:
                # Parse correlation value to provide guidance
                corr_val = self._extract_correlation_value(category, rel_cat)
                if corr_val is not None:
                    if corr_val > 0:
                        guidance = "유사한 응답 수준 예상"
                    else:
                        guidance = "반대 응답 수준 예상"
                    context_parts.append(f"- {rel_cat} {corr_info}: {guidance}")
                else:
                    context_parts.append(f"- {rel_cat} {corr_info}")
        
        if context_parts:
            return "[관련 카테고리 상관관계]\n" + "\n".join(context_parts)
        return ""

    def _get_concise_kg_correlation(self, question: str, category: str) -> str:
        """
        Ultra-concise KG contribution for existing questions: ONE line only.
        Returns correlation direction and related category's response level.
        Designed for Gemma3 12B which gets confused by verbose context.
        """
        if not self.use_kg:
            return ""

        related_cats = self._find_related_categories(question, category)
        if not related_cats:
            return ""

        primary_cat = related_cats[0]
        corr_val = self._extract_correlation_value(category, primary_cat)

        # Get related category's response level from raw history or LTM
        level = self._get_related_cat_level(primary_cat)
        if not level:
            return ""

        if corr_val is not None and corr_val > 0:
            return f"[KG] {primary_cat}({level}) → 유사한 응답 수준 예상"
        elif corr_val is not None and corr_val < 0:
            return f"[KG] {primary_cat}({level}) → 반대 응답 수준 예상"
        else:
            return f"[KG] 관련 카테고리 {primary_cat}: {level} 수준"

    def _get_related_cat_level(self, category: str) -> str:
        """Get the response level for a category from LTM data or raw history."""
        last_values = []

        # Try LTM data first
        patterns = self.ltm_data.get('temporal_patterns', {})
        if patterns:
            for q, pattern in patterns.items():
                if pattern.get('topic') == category:
                    lv = pattern.get('last_value', '')
                    if lv:
                        last_values.append(lv)

        # Fallback: raw history (most recent year)
        if not last_values:
            for year in sorted(self.history.keys(), reverse=True):
                if int(year) >= Config.TARGET_YEAR:
                    continue
                for q, v in self.history[year].items():
                    q_cat = extract_category(q)
                    if q_cat == category and v:
                        last_values.append(v)
                if last_values:
                    break

        if not last_values:
            return ""

        mode_val = Counter(last_values).most_common(1)[0][0]
        return self._classify_response_level(mode_val)

    def _classify_response_level(self, value_text: str) -> str:
        """Classify a response value text into low/mid/high level."""
        if not value_text:
            return "mid"
        # Low: clear negation or absence
        low_keywords = ['전혀', '없다', '않는다', '않다', '하지 않', '않은 편', '거의 없', '거의 하지']
        # High: clear affirmation or frequency
        high_keywords = ['매우', '자주', '항상', '많이', '그런 편이다', '그렇다']
        for kw in low_keywords:
            if kw in value_text:
                return "low"
        for kw in high_keywords:
            if kw in value_text:
                return "high"
        return "mid"

    def _get_response_level_from_pattern(self, pattern: Dict) -> str:
        """Determine response level (low/mid/high) from temporal pattern."""
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
        if not last_value:
            return "mid"
        
        low_keywords = ['전혀', '없다', '않는다', '않다']
        high_keywords = ['매우', '자주', '항상', '많이', '그런 편이다', '그렇다']
        
        for kw in low_keywords:
            if kw in last_value:
                return "low"
        for kw in high_keywords:
            if kw in last_value:
                return "high"
        return "mid"
    
    def _get_related_category_detail(self, question: str, category: str, top_k: int = 3) -> str:
        """
        Build related category context with response statistics.
        Paper Section 2.4: "response statistics (Stability, Mode)" from
        related categories identified via the knowledge graph.

        Provides Mode and Stability summary per related category,
        combined with KG correlation information.

        Args:
            top_k: Number of related categories to include (default 3, use 5 for cold-start)
        """
        if not self.use_kg:
            return ""

        related_cats = self._find_related_categories(question, category)
        if not related_cats:
            return ""

        patterns = self.ltm_data.get('temporal_patterns', {}) if self.use_lte else {}
        context_parts = []

        for rel_cat in related_cats[:top_k]:
            corr_info = self._get_correlation_info(category, rel_cat)

            # Collect response statistics (Stability, Mode) using topic index
            last_values = []
            stabilities = []
            if self.use_lte:
                for q, pattern in self._patterns_by_topic.get(rel_cat, []):
                    last_val = pattern.get('last_value', '')
                    stability = pattern.get('stability', '')
                    if last_val:
                        last_values.append(last_val)
                    if stability:
                        stabilities.append(stability)

            if last_values:
                # Mode (most frequent response) - Paper Section 2.4
                mode_val = Counter(last_values).most_common(1)[0]
                # Stability summary
                stab_counter = Counter(stabilities)
                dominant_stab = stab_counter.most_common(1)[0][0] if stab_counter else "unknown"

                header = f"- {rel_cat}"
                if corr_info:
                    header += f" {corr_info}"
                header += f" | Mode: {mode_val[0]} ({mode_val[1]}/{len(last_values)}문항), Stability: {dominant_stab}"
                context_parts.append(header)
            elif corr_info:
                corr_val = self._extract_correlation_value(category, rel_cat)
                guidance = ""
                if corr_val is not None:
                    guidance = ": 유사한 응답 수준 예상" if corr_val > 0 else ": 반대 응답 수준 예상"
                context_parts.append(f"- {rel_cat} {corr_info}{guidance}")

        if context_parts:
            return "[관련 카테고리 응답 통계]\n" + "\n".join(context_parts)
        return ""

    def _get_category_summaries(self, related_cats: List[str]) -> List[str]:
        """
        Get category summaries (Mode, Stability) for cold-start prediction.
        Paper Figure 7: category_summaries section.
        """
        summaries = []
        profiles = self.ltm_data.get('thematic_profiles', {})
        patterns = self.ltm_data.get('temporal_patterns', {})

        for cat in related_cats[:3]:
            if cat not in profiles:
                continue

            prof = profiles[cat]
            stab_score = prof.get('stability_score', 0.5)

            if stab_score >= 0.8:
                stab_desc = "highly_stable"
            elif stab_score >= 0.6:
                stab_desc = "stable"
            else:
                stab_desc = "variable"

            # Find mode value for this category
            mode_values = []
            for q, p in patterns.items():
                if p.get('topic') == cat:
                    mode_values.append(p.get('mode', ''))

            if mode_values:
                mode_counter = Counter(mode_values)
                common_mode = mode_counter.most_common(1)[0][0]
                summaries.append(f"- [{cat}] {stab_desc}, Mode: '{common_mode}'")
            else:
                summaries.append(f"- [{cat}] {stab_desc}")

        return summaries

    def _get_strong_correlations(self, category: str) -> List[str]:
        """
        Get strong correlations for cold-start prediction.
        Paper Figure 7: strong_correlations section with explicit interpretation.
        Falls back to semantic similarity when behavioral data is missing.
        """
        correlations = []
        corr_data = _load_behavioral_correlation()

        relationships = corr_data.get('category_relationships', {}) if corr_data else {}

        if category in relationships:
            for item in relationships[category][:3]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    corr_cat, corr_val = item[0], item[1]
                    direction = "positive" if corr_val > 0 else "negative"
                    strength = "strong" if abs(corr_val) >= 0.5 else "moderate"

                    if corr_val > 0:
                        interp = f"high '{corr_cat}' -> high '{category}'"
                    else:
                        interp = f"high '{corr_cat}' -> low '{category}'"

                    correlations.append(
                        f"- '{category}' <-> '{corr_cat}' (Corr: {corr_val:+.2f}, {strength} {direction}): {interp}"
                    )
        else:
            # Fallback: use semantic similarity for cold-start categories
            sim_data = _load_category_similarity()
            if sim_data:
                sim_rels = sim_data.get('category_relationships', {}).get(category, [])
                for item in sim_rels[:3]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        sim_cat, sim_val = item[0], item[1]
                        # Semantic similarity implies positive correlation
                        interp = f"similar to '{sim_cat}' -> expect similar response level"
                        correlations.append(
                            f"- '{category}' ~ '{sim_cat}' (Similarity: {sim_val:.2f}): {interp}"
                        )

        return correlations

    def _get_2022_sudden_changes(self, category: str, related_cats: List[str]) -> List[str]:
        """
        Get 2022 sudden changes in target and related categories.
        Paper Figure 7: sudden_changes section.
        """
        changes = []
        target_cats = [category] + related_cats

        for q, shift in self.ltm_data.get('sudden_shifts', {}).items():
            if shift.get('year') == 2022:
                q_cat = extract_category(q)
                if q_cat in target_cats:
                    z_score = shift.get('z_score', 'N/A')
                    z_str = f"Z-score: {z_score}" if isinstance(z_score, (int, float)) else ""
                    changes.append(
                        f"- [{q_cat}] {shift.get('from_value', '')} -> {shift.get('to_value', '')} {z_str}"
                    )

        return changes[:5]

    def _infer_cold_start_level(self, question: str, category: str,
                                options: Dict[str, str]) -> str:
        """
        KG-based response level inference for cold-start questions.
        Paper Section 2.4: Uses KG correlation edges to identify related categories
        and provide their response statistics as alternative context.

        Key design: Instead of providing raw response level (which misleads the LLM
        for negative-behavior items), provide:
        1. The correlation relationship (which category is related)
        2. Whether this student's pattern is low/typical (aligns with majority)
           or high (potential outlier)
        3. The option 1 frequency in related category (concrete statistic)

        This gives the LLM useful structural information without misleading it.
        """
        if not self.use_kg:
            return ""

        related_cats = self._find_related_categories(question, category)
        if not related_cats:
            return ""

        # Use the most similar category (first one from KG)
        primary_cat = related_cats[0]

        # Get correlation value if available
        corr_val = self._extract_correlation_value(category, primary_cat)

        # Collect response values for the related category
        last_values = []

        # Try LTM temporal patterns first (available when use_lte=True)
        patterns = self.ltm_data.get('temporal_patterns', {})
        if patterns:
            for q, pattern in patterns.items():
                if pattern.get('topic') == primary_cat:
                    lv = pattern.get('last_value', '')
                    if lv:
                        last_values.append(lv)

        # Fallback: use raw history data (works even without LTE)
        if not last_values:
            for year in sorted(self.history.keys(), reverse=True):
                if int(year) >= Config.TARGET_YEAR:
                    continue
                for q, v in self.history[year].items():
                    q_cat = extract_category(q)
                    if q_cat == primary_cat and v:
                        last_values.append(v)
                if last_values:
                    break  # Use most recent year's data

        if not last_values:
            return ""

        # Count how many responses are option 1 (lowest) in the related category
        low_keywords = ['전혀', '없다', '않는다', '않다', '하지 않', '않은 편', '거의 없', '거의 하지']
        option1_count = 0
        for v in last_values:
            if any(kw in v for kw in low_keywords):
                option1_count += 1
        option1_ratio = option1_count / len(last_values) if last_values else 0

        # Build KG context: correlation + student's related-category pattern
        parts = [f"[KG 분석] 관련 카테고리: {primary_cat}"]
        if corr_val is not None:
            parts.append(f"상관계수: r={corr_val:.2f} (양의 상관: 유사한 응답 경향)")
        parts.append(f"이 학생의 {primary_cat} 응답 중 최저 수준 비율: {option1_ratio:.0%} ({option1_count}/{len(last_values)}건)")

        return "\n".join(parts)

    def _get_pattern_description(self, question: str) -> str:
        """
        Get compact pattern description combining stability, level, and trend.
        Format: "매우안정-low, 상승추세 (최근: '그렇지 않은 편이다')"
        """
        if not self.use_lte:
            return ""

        pattern = self._find_in_dict(question, self.ltm_data.get('temporal_patterns', {}))
        if not pattern:
            return ""

        # Stability
        stability = pattern.get('stability', 'unknown')
        stab_map = {
            'constant': '완전일정',
            'highly_stable': '매우안정',
            'stable': '안정',
            'variable': '변동',
        }
        stab_desc = stab_map.get(stability, stability)

        # Response level
        level = self._get_response_level_from_pattern(pattern)

        # Trend direction
        trend = pattern.get('trend') if pattern else None
        dir_desc = ""
        if trend and isinstance(trend, dict):
            direction = trend.get('direction', '')
            if direction == 'increasing':
                dir_desc = ", 상승추세"
            elif direction == 'decreasing':
                dir_desc = ", 하강추세"

        # Last value
        last_value = pattern.get('last_value', '')
        last_part = f" (최근: '{last_value}')" if last_value else ""

        return f"{stab_desc}-{level}{dir_desc}{last_part}"

    def _extract_correlation_value(self, cat1: str, cat2: str) -> Optional[float]:
        """Extract numeric correlation value between two categories."""
        # Student-specific KG
        kg_relationships = self.kg_data.get('category_relationships', {})
        for cat in [cat1, cat2]:
            if cat in kg_relationships:
                for item in kg_relationships[cat]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        other_cat = cat2 if cat == cat1 else cat1
                        if item[0] == other_cat:
                            return float(item[1])
        
        # Global behavioral correlation
        behavioral_corr = _load_behavioral_correlation()
        if behavioral_corr:
            relationships = behavioral_corr.get('category_relationships', {})
            for cat in [cat1, cat2]:
                if cat in relationships:
                    for item in relationships[cat]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            other_cat = cat2 if cat == cat1 else cat1
                            if item[0] == other_cat:
                                return float(item[1])
        
        return None
    
    def _find_related_categories(self, question: str, category: str) -> List[str]:
        """
        Find related categories using KG graph traversal.

        Priority:
        1. Actual graph query via MirrorKnowledgeGraph.get_related_domains()
        2. Student-specific KG relationships
        3. Global behavioral correlation
        4. Semantic similarity

        Returns up to 5 related categories (callers slice as needed).
        """
        if category in self._related_categories_cache:
            return self._related_categories_cache[category]

        related = []

        # Actual graph query
        if HAS_KG_CLASS and self.kg_data:
            kg_graph = _load_knowledge_graph(self.student_id, self.kg_data)
            if kg_graph:
                try:
                    graph_related = kg_graph.get_related_domains(category, top_k=7)
                    for rel_cat, weight in graph_related:
                        if rel_cat != category:
                            related.append(rel_cat)
                    if related:
                        if self.debug:
                            print(f"[KG] Graph query found {len(related)} related categories for {category}")
                        self._related_categories_cache[category] = related[:5]
                        return related[:5]
                except Exception as e:
                    if self.debug:
                        print(f"[KG] Graph query failed: {e}")

        # Student-specific KG relationships (dict fallback)
        kg_relationships = self.kg_data.get('category_relationships', {})
        if category in kg_relationships:
            for item in kg_relationships[category][:7]:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    if item[0] != category:
                        related.append(item[0])

        # Global behavioral correlation
        if not related:
            behavioral_corr = _load_behavioral_correlation()
            if behavioral_corr:
                relationships = behavioral_corr.get('category_relationships', {})
                if category in relationships:
                    for item in relationships[category][:7]:
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            if item[0] != category:
                                related.append(item[0])

        # Semantic similarity
        if not related:
            category_sim = _load_category_similarity()
            if category_sim:
                sim_rels = category_sim.get('category_relationships', {}).get(category, [])
                for item in sim_rels[:7]:
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        if item[0] != category:
                            related.append(item[0])

        self._related_categories_cache[category] = related[:5]
        return related[:5]
    
    def _get_correlation_info(self, cat1: str, cat2: str) -> str:
        """Get correlation info between two categories"""
        # Student-specific KG
        kg_relationships = self.kg_data.get('category_relationships', {})
        for cat in [cat1, cat2]:
            if cat in kg_relationships:
                for item in kg_relationships[cat]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        other_cat = cat2 if cat == cat1 else cat1
                        if item[0] == other_cat:
                            corr = item[1]
                            sign = "+" if corr > 0 else ""
                            return f"(r={sign}{corr:.2f})"
        
        # Global behavioral correlation
        behavioral_corr = _load_behavioral_correlation()
        if not behavioral_corr:
            return ""
        
        relationships = behavioral_corr.get('category_relationships', {})
        
        for cat in [cat1, cat2]:
            if cat in relationships:
                for item in relationships[cat]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        other_cat = cat2 if cat == cat1 else cat1
                        if item[0] == other_cat:
                            corr = item[1]
                            sign = "+" if corr > 0 else ""
                            return f"(r={sign}{corr:.2f})"
        
        return ""
    
    # Consistency-aware Prediction
    
    def _build_context(self, question: str, options: Dict[str, str]) -> Dict:
        """
        Build dynamic context CTX_{u,q} = (q, Y_q, E_{u,q}, P_{u,q}, C_q)
        
        Component isolation (following paper Table 5 ablation):
        - Base history: ALWAYS provided (all methods build on LLM-only baseline)
        - E_{u,q}: Additional retrieved evidence (RER component)
        - P_{u,q}: Enriched persona profiles (LTE component)
        - C_q}: KG constraints (KG component)
        
        Key insight from paper:
        - LLM-only: raw history concatenation (baseline)
        - +RER: LLM-only + semantic evidence retrieval
        - +RER+LTE: RER + enriched persona (static/dynamic/narrative)
        - +RER+KG: RER + consistency constraints
        - MIRROR: RER + LTE + KG
        """
        category = extract_category(question)
        
        # Base: Raw history for this question (ALWAYS provided - basis of LLM-only)
        raw_hist = self._get_question_history(question)
        is_new_question = len(raw_hist) == 0
        
        # Base yearly trace from raw history (provided to all methods)
        base_yearly_trace = ""
        last_value = None
        if raw_hist:
            base_yearly_trace = " -> ".join([f"{h['year']}:{h['value']}" for h in raw_hist])
            last_value = raw_hist[-1]['value']
        
        # E_{u,q}: Retrieved evidence (RER component only)
        evidence = self._retrieve_evidence(question, k=5) if self.use_rer else ""
        
        # P_{u,q}: Enriched Persona profiles (LTE component only)
        # These are ADDITIONAL to base history, not replacement
        static_persona = self._get_static_persona() if self.use_lte else ""  # LTE only
        domain_profile = self._get_domain_profile(category) if self.use_lte else ""  # LTE only
        change_warning = self._get_change_profile(question) if self.use_lte else ""  # LTE only
        narrative = self._get_overall_narrative() if self.use_lte else ""  # LTE only
        
        # Stability analysis is LTE enrichment
        stability = None
        pattern = None
        if self.use_lte:
            # LTE: get enriched pattern data with stability analysis
            pattern = self._find_in_dict(question, self.ltm_data.get('temporal_patterns', {}))
            if pattern:
                stability = pattern.get('stability', 'unknown')
                if pattern.get('last_value'):
                    last_value = pattern['last_value']
            elif raw_hist:
                # Compute stability from raw history
                values = [h['value'] for h in raw_hist]
                unique = len(set(values))
                if unique == 1:
                    stability = 'constant'
                elif unique <= 2 and len(values) >= 3:
                    stability = 'stable'
                else:
                    stability = 'variable'
        
        # C_q: KG constraints (KG component only)
        kg_constraints = self._get_kg_constraints(question, category) if self.use_kg else ""
        related_context = self._get_related_category_context(question, category) if self.use_kg else ""

        # KG valid option range constraint
        kg_valid_options = ""
        if self.use_kg:
            kg_valid_options = self._get_kg_valid_option_range(question)

        return {
            'category': category,
            'is_new_question': is_new_question,
            'evidence': evidence,
            'static_persona': static_persona,
            'domain_profile': domain_profile,
            'base_yearly_trace': base_yearly_trace,  # Always provided (LLM-only baseline)
            'pattern': pattern,
            'change_warning': change_warning,
            'narrative': narrative,
            'kg_constraints': kg_constraints,
            'related_context': related_context,
            'kg_valid_options': kg_valid_options,  # KG valid option range
            'stability': stability,  # LTE only
            'last_value': last_value,
        }
    
    def predict(self, question: str, options: Dict[str, str]) -> Tuple[str, str, Dict]:
        """
        Single-step constrained inference.
        ŷ^(T+1)_u = g(q, Y_q, E_{u,q}, P_{u,q}, C_q)
        """
        ctx = self._build_context(question, options)
        
        if self.debug:
            print(f"\n[MIRROR] Question: {question[:50]}...")
            print(f"[MIRROR] Category: {ctx['category']}, New: {ctx['is_new_question']}")
            print(f"[MIRROR] Evidence chars: {len(ctx['evidence'])}")
            print(f"[MIRROR] Stability: {ctx['stability']}")
            print(f"[MIRROR] Has narrative: {bool(ctx['narrative'])}")
            print(f"[MIRROR] Has KG constraints: {bool(ctx['kg_constraints'])}")
        
        if ctx['is_new_question']:
            return self._predict_cold_start(question, options, ctx)
        else:
            return self._predict_existing(question, options, ctx)
    
    def _predict_existing(self, question: str, options: Dict[str, str],
                          ctx: Dict) -> Tuple[str, str, Dict]:
        """
        Prediction for existing questions (has history).
        Paper: ŷ = g(q, Y_q, E_{u,q}, P_{u,q}, C_q)
        """
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

        prompt_parts = []
        prompt_parts.append("학생의 과거 응답 추세를 이어서 2023년 응답을 예측하세요. 번호만 출력하세요.")

        # [LTE] P^static + narrative
        if ctx['static_persona']:
            prompt_parts.append(ctx['static_persona'])
        if ctx['narrative']:
            prompt_parts.append(ctx['narrative'][:150])
        if ctx['change_warning']:
            prompt_parts.append(ctx['change_warning'])

        # [BASE] Y_q: Time-series trace (always provided)
        if ctx['base_yearly_trace']:
            prompt_parts.append(ctx['base_yearly_trace'])

        # [LTE] Pattern description
        if self.use_lte:
            pattern_desc = self._get_pattern_description(question)
            if pattern_desc:
                prompt_parts.append(pattern_desc)

        # [KG] Related category detail (Mode/Stability)
        if self.use_kg:
            related_detail = self._get_related_category_detail(question, ctx['category'])
            if related_detail:
                prompt_parts.append(related_detail)

        # [RER] Retrieved evidence
        if ctx['evidence']:
            prompt_parts.append(ctx['evidence'][:300])

        # [KG] Valid option range
        if ctx.get('kg_valid_options'):
            prompt_parts.append(ctx['kg_valid_options'])

        # Question + answer
        prompt_parts.append(f"질문: {question}")
        prompt_parts.append(options_str)
        prompt_parts.append("답:")

        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm.invoke(prompt)
            pred = clean_llm_output(response.content)

            # Store full LLM response as explanation (paper Figure 2)
            ctx['explanation'] = response.content.strip()

            if self.debug:
                print(f"[MIRROR] LLM output: {response.content[:100]}")
                print(f"[MIRROR] Cleaned: {pred}")

            if pred != "0" and pred in options:
                method = f"MIRROR:existing:{ctx['stability'] or 'unknown'}"
                if ctx['change_warning']:
                    method += ":shift"
                return pred, method, ctx
        except Exception as e:
            if self.debug:
                print(f"[MIRROR] LLM error: {e}")

        # Fallback (respects component flags)
        return self._fallback_prediction(question, options, ctx)

    def _get_related_category_traces(self, n_cats: int = 2, n_traces: int = 5) -> list:
        """
        RER cold-start: Get raw answer traces from related categories.
        Returns list of formatted trace strings (e.g., "2022년 [공격성]: 그렇지 않은 편이다").
        """
        traces = []
        for rel_cat in RELATED_CATEGORY_PRIORITY[:n_cats]:
            cat_traces = []
            for year in sorted(self.history.keys(), reverse=True):
                if int(year) >= Config.TARGET_YEAR:
                    continue
                for q, v in self.history[year].items():
                    if extract_category(q) == rel_cat and v:
                        cat_traces.append(f"{year}년 [{rel_cat}]: {v}")
                if cat_traces:
                    break
            traces.extend(cat_traces[:n_traces])
        return traces

    def _predict_cold_start(self, question: str, options: Dict[str, str],
                            ctx: Dict) -> Tuple[str, str, Dict]:
        """
        Prediction for cold-start questions (no history).
        Paper: ŷ = g(q, E_{u,q}, P_{u,q}, C_q)

        Cold-start strategy: Use raw traces from related categories as
        pseudo-history. The LLM infers the response level from related
        behavioral patterns, formatted like existing question prompts.
        """
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

        prompt_parts = []

        # [LTE] P^static + narrative (when available)
        if ctx['static_persona']:
            prompt_parts.append(ctx['static_persona'])
        if ctx['narrative']:
            prompt_parts.append(ctx['narrative'][:150])

        # [KG] Related category detail (Mode/Stability) from KG
        if self.use_kg:
            related_detail = self._get_related_category_detail(question, ctx['category'])
            if related_detail:
                prompt_parts.append(related_detail)
            elif ctx['kg_constraints']:
                prompt_parts.append(ctx['kg_constraints'])

        # [RER] Core: Related category traces as pseudo-history
        # Raw traces from related categories give the LLM direct behavioral data
        # to infer the student's likely response level for the new domain.
        traces = self._get_related_category_traces(n_cats=2, n_traces=5)
        if traces:
            prompt_parts.extend(traces[:5])

        # [LTE] Domain profile
        if ctx['domain_profile']:
            prompt_parts.append(ctx['domain_profile'])

        # [KG] Valid option range
        if ctx.get('kg_valid_options'):
            prompt_parts.append(ctx['kg_valid_options'])

        # Instruction + question (same style as existing question prompts)
        prompt_parts.append("학생의 과거 응답 추세를 이어서 2023년 응답을 예측하세요. 번호만 출력하세요.")
        prompt_parts.append(f"질문: {question}")
        prompt_parts.append(options_str)
        prompt_parts.append("답:")

        prompt = "\n".join(prompt_parts)

        try:
            response = self.llm.invoke(prompt)
            pred = clean_llm_output(response.content)

            # Store full LLM response as explanation (paper Figure 2)
            ctx['explanation'] = response.content.strip()

            if self.debug:
                print(f"[MIRROR] Cold-start LLM output: {response.content[:100]}")
                print(f"[MIRROR] Cleaned: {pred}")

            if pred != "0" and pred in options:
                return pred, f"MIRROR:cold_start:{ctx['category']}", ctx
        except Exception as e:
            if self.debug:
                print(f"[MIRROR] LLM error: {e}")

        # Fallback for cold-start (respects component flags)
        return self._fallback_cold_start(question, options, ctx)
    
    def _fallback_prediction(self, question: str, options: Dict[str, str],
                             ctx: Dict) -> Tuple[str, str, Dict]:
        """
        Fallback when LLM fails.
        Uses last observed value or mode of historical responses.
        """
        if ctx.get('last_value'):
            pred_key = self._find_option_key(ctx['last_value'], options)
            if pred_key != "0":
                return pred_key, "MIRROR:fallback:last_value", ctx
        
        hist = self._get_question_history(question)
        if hist:
            values = [h['value'] for h in hist]
            mode_value = Counter(values).most_common(1)[0][0]
            pred_key = self._find_option_key(mode_value, options)
            if pred_key != "0":
                return pred_key, "MIRROR:fallback:mode", ctx
        
        pred = get_fallback_prediction(hist, options)
        return pred, "MIRROR:fallback:final", ctx
    
    def _fallback_cold_start(self, question: str, options: Dict[str, str],
                             ctx: Dict) -> Tuple[str, str, Dict]:
        """
        Fallback for cold-start questions.
        Use related category response levels to infer the appropriate fallback.
        """
        option_keys = sorted(options.keys())

        # Check related category response levels from raw history
        # This works for ALL configurations (RER-only, RER+LTE, RER+KG, MIRROR)
        category = ctx.get('category', '')

        # Method 1: LTE temporal patterns (richest data)
        if self.use_lte:
            related_cats = self._find_related_categories(question, category)
            levels = []
            for rel_cat in related_cats[:5]:
                for q, pattern in self._patterns_by_topic.get(rel_cat, []):
                    level = self._get_response_level_from_pattern(pattern)
                    levels.append(level)

            if levels:
                low_count = levels.count('low')
                high_count = levels.count('high')

                if low_count > high_count:
                    return option_keys[0], "MIRROR:cold_start:fallback_low", ctx
                elif high_count > low_count:
                    return option_keys[-1], "MIRROR:cold_start:fallback_high", ctx

        # Method 2: Raw history from related categories (works for RER-only)
        if RELATED_CATEGORY_PRIORITY:
            low_keywords = ['전혀', '없다', '않는다', '않다', '하지 않', '않은 편', '거의 없', '거의 하지']
            all_values = []
            for rel_cat in RELATED_CATEGORY_PRIORITY[:3]:
                for year in sorted(self.history.keys(), reverse=True):
                    if int(year) >= Config.TARGET_YEAR:
                        continue
                    for q, v in self.history[year].items():
                        if extract_category(q) == rel_cat and v:
                            all_values.append(v)
                    if all_values:
                        break

            if all_values:
                low_count = sum(1 for v in all_values if any(kw in v for kw in low_keywords))
                low_ratio = low_count / len(all_values)
                if low_ratio >= 0.5:
                    return option_keys[0], "MIRROR:cold_start:fallback_low", ctx

        # Default: use first option (lowest) for frequency scales, median otherwise
        option_type = detect_option_type(options)
        if option_type == 'frequency':
            return option_keys[0], "MIRROR:cold_start:fallback_freq_low", ctx
        mid_idx = len(option_keys) // 2
        return option_keys[mid_idx] if option_keys else "1", "MIRROR:cold_start:fallback_median", ctx
    
    # Utility methods

    def _normalize_for_match(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'\s+', '', str(text)).lower()
    
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
    
    def _get_question_history(self, question: str) -> List[Dict]:
        """Get historical responses for a question from raw history data."""
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
    
    def _build_prompt(self, question: str, options: Dict[str, str]) -> Tuple[str, Dict]:
        """Build prompt string and context for a question (no LLM call)."""
        ctx = self._build_context(question, options)
        options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

        if ctx['is_new_question']:
            prompt_parts = []
            # [LTE] P^static + narrative
            if ctx['static_persona']:
                prompt_parts.append(ctx['static_persona'])
            if ctx['narrative']:
                prompt_parts.append(ctx['narrative'][:150])
            # [KG] Related category detail
            if self.use_kg:
                related_detail = self._get_related_category_detail(question, ctx['category'])
                if related_detail:
                    prompt_parts.append(related_detail)
                elif ctx['kg_constraints']:
                    prompt_parts.append(ctx['kg_constraints'])
            # [RER] Core: Related category traces as pseudo-history
            traces = self._get_related_category_traces(n_cats=2, n_traces=5)
            if traces:
                prompt_parts.extend(traces[:5])
            # [LTE] Domain profile
            if ctx['domain_profile']:
                prompt_parts.append(ctx['domain_profile'])
            if ctx.get('kg_valid_options'):
                prompt_parts.append(ctx['kg_valid_options'])
            # Instruction (same style as existing question prompts)
            prompt_parts.append("학생의 과거 응답 추세를 이어서 2023년 응답을 예측하세요. 번호만 출력하세요.")
            prompt_parts.append(f"질문: {question}")
            prompt_parts.append(options_str)
            prompt_parts.append("답:")
        else:
            prompt_parts = []
            # Task instruction for existing questions
            prompt_parts.append("학생의 과거 응답 추세를 이어서 2023년 응답을 예측하세요. 번호만 출력하세요.")
            if ctx['static_persona']:
                prompt_parts.append(ctx['static_persona'])
            if ctx['narrative']:
                prompt_parts.append(ctx['narrative'][:150])
            if ctx['change_warning']:
                prompt_parts.append(ctx['change_warning'])
            if ctx['base_yearly_trace']:
                prompt_parts.append(ctx['base_yearly_trace'])
            if self.use_lte:
                pattern_desc = self._get_pattern_description(question)
                if pattern_desc:
                    prompt_parts.append(pattern_desc)
            if self.use_kg:
                related_detail = self._get_related_category_detail(question, ctx['category'])
                if related_detail:
                    prompt_parts.append(related_detail)
            if ctx['evidence']:
                prompt_parts.append(ctx['evidence'][:300])
            if ctx.get('kg_valid_options'):
                prompt_parts.append(ctx['kg_valid_options'])
            prompt_parts.append(f"질문: {question}")
            prompt_parts.append(options_str)
            prompt_parts.append("답:")

        return "\n".join(prompt_parts), ctx

    def predict_batch(self, tasks: List[Dict], verbose: bool = False) -> Tuple[List[str], List[str]]:
        import concurrent.futures
        from tqdm import tqdm

        desc = f"[MIRROR] {self.student_id} (RER={self.use_rer},LTE={self.use_lte},KG={self.use_kg})"

        # Phase 1: Build all prompts (CPU, fast)
        prompts = []
        contexts = []
        for task in tasks:
            prompt, ctx = self._build_prompt(task['question'], task['options'])
            prompts.append(prompt)
            contexts.append(ctx)

        # Phase 2: Send all LLM requests concurrently
        def call_llm(idx):
            try:
                response = self.llm.invoke(prompts[idx])
                return idx, response.content
            except Exception as e:
                return idx, ""

        results = [""] * len(tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(call_llm, i): i for i in range(len(tasks))}
            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(tasks), desc=desc, leave=False):
                idx, content = future.result()
                results[idx] = content

        # Phase 3: Parse results
        preds, reasons = [], []
        for i, task in enumerate(tasks):
            ctx = contexts[i]
            content = results[i]
            pred = clean_llm_output(content)
            ctx['explanation'] = content.strip()

            if pred != "0" and pred in task['options']:
                if ctx['is_new_question']:
                    reason = f"MIRROR:cold_start:{ctx['category']}"
                else:
                    reason = f"MIRROR:existing:{ctx['stability'] or 'unknown'}"
            else:
                # Fallback
                if ctx['is_new_question']:
                    pred, reason, ctx = self._fallback_cold_start(
                        task['question'], task['options'], ctx)
                else:
                    pred, reason, ctx = self._fallback_prediction(
                        task['question'], task['options'], ctx)

            preds.append(pred)
            explanation = ctx.get('explanation', '')
            reasons.append(f"{reason} | {explanation}" if explanation else reason)

        return preds, reasons


# Factory functions for different configurations
def create_predictor(student_id: str, 
                     method: str = "MIRROR",
                     exclude_target: bool = False, 
                     exclude_partial: bool = False,
                     tool_set: str = 'full',
                     debug: bool = False) -> MirrorPredictor:
    """
    Create predictor with specific configuration.
    
    Methods
    - LLM_only: No components (baseline)
    - RER: Only evidence retrieval
    - RER_LTE: Evidence + Longitudinal trends
    - RER_KG: Evidence + Knowledge graph  
    - MIRROR: All components (RER + LTE + KG)
    
    Component Isolation Configurations
    - LLM_only: use_rer=False, use_lte=False, use_kg=False
    - RER: use_rer=True, use_lte=False, use_kg=False
    - RER_LTE: use_rer=True, use_lte=True, use_kg=False
    - RER_KG: use_rer=True, use_lte=False, use_kg=True
    - MIRROR: use_rer=True, use_lte=True, use_kg=True
    """
    configs = {
        'LLM_only': {'use_rer': False, 'use_lte': False, 'use_kg': False},
        'RER': {'use_rer': True, 'use_lte': False, 'use_kg': False},
        'RER_LTE': {'use_rer': True, 'use_lte': True, 'use_kg': False},
        'RER_KG': {'use_rer': True, 'use_lte': False, 'use_kg': True},
        'MIRROR': {'use_rer': True, 'use_lte': True, 'use_kg': True},
        # Legacy mappings (for backward compatibility)
        'rag_only': {'use_rer': True, 'use_lte': False, 'use_kg': False},   # RER
        'rag_stm_ltm': {'use_rer': True, 'use_lte': True, 'use_kg': False}, # RER_LTE
        'full': {'use_rer': True, 'use_lte': True, 'use_kg': True},         # MIRROR
    }
    
    config = configs.get(method, configs['MIRROR'])
    
    if debug:
        print(f"[create_predictor] Method: {method}")
        print(f"[create_predictor] Config: {config}")
    
    return MirrorPredictor(
        student_id=student_id,
        exclude_target=exclude_target,
        exclude_partial=exclude_partial,
        use_rer=config['use_rer'],
        use_lte=config['use_lte'],
        use_kg=config['use_kg'],
        tool_set=tool_set,
        debug=debug
    )
