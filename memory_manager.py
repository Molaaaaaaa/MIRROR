import os
import json
import re
from typing import Dict, List, Optional
from config import Config


def extract_topic(question_key: str) -> str:
    """extract topic from question key"""
    match = re.match(r'\[([^\]]+)\]', question_key)
    if match:
        return match.group(1).strip()
    if '-' in question_key:
        return question_key.split('-')[0].strip()
    return question_key.strip()


def normalize_question(question: str) -> str:
    """normalize question text for matching"""
    if not question:
        return ""
    text = re.sub(r'\s+', ' ', question.strip())
    text = text.replace('，', ',').replace('．', '.').replace('"', '"').replace('"', '"')
    return text


class MemoryManager:
    def __init__(self, student_id: str, ltm_dir: str = None):
        self.student_id = student_id
        self.ltm_dir = ltm_dir or Config.MIRROR_MEMORY_DIR
        
        self.ltm_data = self._load_ltm()
        
        self._norm_index = self._build_norm_index()
    
    def _load_ltm(self) -> Dict:
        for suffix in ['full_pipeline', 'rich', '']:
            if suffix:
                path = os.path.join(self.ltm_dir, f"{self.student_id}_ltm_{suffix}.json")
            else:
                path = os.path.join(self.ltm_dir, f"{self.student_id}_ltm.json")
            
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return {
            'temporal_patterns': {},
            'thematic_profiles': {},
            'overall_narrative': '',
            'yearly_changes': {},
            'prediction_hints': {},
            'sudden_shifts': {},
            'data_quality': {}
        }
    
    def _build_norm_index(self) -> Dict[str, str]:
        index = {}
        for key in self.ltm_data.get('temporal_patterns', {}).keys():
            norm = normalize_question(key)
            index[norm] = key
        return index
    
    def _find_key(self, question: str) -> Optional[str]:
        patterns = self.ltm_data.get('temporal_patterns', {})
        
        if question in patterns:
            return question
        
        norm = normalize_question(question)
        if norm in self._norm_index:
            return self._norm_index[norm]
        
        for key in patterns.keys():
            if question in key or key in question:
                return key
        
        return None
    
    @property
    def temporal_patterns(self) -> Dict:
        return self.ltm_data.get('temporal_patterns', {})
    
    @property
    def overall_narrative(self) -> str:
        return self.ltm_data.get('overall_narrative', '')
    
    @property
    def yearly_changes(self) -> Dict:
        return self.ltm_data.get('yearly_changes', {})
    
    @property
    def prediction_hints(self) -> Dict:
        return self.ltm_data.get('prediction_hints', {})
    
    @property
    def sudden_shifts(self) -> Dict:
        return self.ltm_data.get('sudden_shifts', {})
    
    @property
    def data_quality(self) -> Dict:
        return self.ltm_data.get('data_quality', {})
    
    def get_time_series(self, question: str) -> str:
        key = self._find_key(question)
        if not key:
            return "No historical data"
        
        pattern = self.temporal_patterns.get(key, {})
        series = pattern.get('series', [])
        
        if not series:
            return "No time series data"
        
        trace = " -> ".join([f"{s['year']}:{s['value']}" for s in series])
        
        stability = pattern.get('stability', 'unknown')
        last_value = pattern.get('last_value', '')
        
        return f"{trace} | [{stability}] recent: '{last_value}'"
    
    def get_question_history(self, question: str) -> List[Dict]:
        key = self._find_key(question)
        if not key:
            return []
        
        pattern = self.temporal_patterns.get(key, {})
        return pattern.get('series', [])
    
    def get_prediction_hint(self, question: str) -> Optional[Dict]:
        key = self._find_key(question)
        if key:
            return self.prediction_hints.get(key)
        
        for k, hint in self.prediction_hints.items():
            if question in k or k in question:
                return hint
        
        return None
    
    def get_sudden_shift(self, question: str) -> Optional[Dict]:
        key = self._find_key(question)
        if key:
            return self.sudden_shifts.get(key)
        
        for k, shift in self.sudden_shifts.items():
            if question in k or k in question:
                return shift
        
        return None
    
    def get_related_questions(self, question: str, max_count: int = 3) -> List[Dict]:
        topic = extract_topic(question)
        related = []
        
        for q, pattern in self.temporal_patterns.items():
            if q == question:
                continue
            if pattern.get('topic') == topic:
                related.append({
                    'question': q,
                    'last_value': pattern.get('last_value', ''),
                    'stability': pattern.get('stability', ''),
                    'has_shift': pattern.get('has_sudden_shift', False)
                })
                if len(related) >= max_count:
                    break
        
        return related
    
    def get_narrative_context(self) -> str:
        narrative = self.overall_narrative
        
        if isinstance(narrative, str) and narrative:
            return narrative[:1000]
        
        if isinstance(narrative, list):
            parts = []
            for n in narrative:
                if isinstance(n, dict):
                    year = n.get('year', '')
                    summary = n.get('summary', '')
                    parts.append(f"[{year}] {summary}")
            return "\n".join(parts)[:1000]
        
        return "No growth narrative"
    
    def get_stm_context(self, question: str) -> str:
        trace = self.get_time_series(question)
        
        related = self.get_related_questions(question)
        if related:
            related_str = "\n".join([
                f"- {r['question'][:40]}...: '{r['last_value']}' [{r['stability']}]"
                for r in related
            ])
        else:
            related_str = "No related questions"
        
        return f"[Time Series]\n{trace}\n\n[Related Questions]\n{related_str}"
    
    def save(self):
        os.makedirs(self.ltm_dir, exist_ok=True)
        path = os.path.join(self.ltm_dir, f"{self.student_id}_ltm_full_pipeline.json")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.ltm_data, f, ensure_ascii=False, indent=2)
