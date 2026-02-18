"""
Graph Structure:
- Nodes: Domain(Topic), Question, Option
- Edges: 
  - Inclusion (Schema): domain -> question -> option
  - Association: correlation-based, semantic similarity-based

This module builds a proper graph structure using NetworkX.
"""
import os
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[Warning] NetworkX not installed. Using dict-based fallback.")

from langchain_ollama import ChatOllama

from config import Config
from utils import load_student_data, get_all_student_ids, extract_category
from data_constants import (
    RESPONSE_SCALE,
    FREQUENCY_KEYWORDS,
    AGREEMENT_KEYWORDS,
)


def get_numeric_value(response: str) -> Optional[float]:
    if response in RESPONSE_SCALE:
        return RESPONSE_SCALE[response]
    try:
        return float(response)
    except:
        return None


def calculate_trend(values: List[float]) -> str:
    """
    Calculate trend direction from a time series of values.
    
    Uses normalized slope to handle different value scales (e.g., 1-4 vs 1-7 Likert).
    The slope is normalized by the value range, so threshold is scale-independent.
    
    Returns:
        'insufficient': Less than 2 data points
        'stable': Normalized annual change < 10% of value range
        'increasing': Positive trend
        'decreasing': Negative trend
    """
    if len(values) < 2:
        return 'insufficient'
    
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    slope = numerator / denominator if denominator != 0 else 0
    
    # Normalize slope by value range for scale-independent threshold
    value_range = max(values) - min(values)
    if value_range == 0:
        # All values identical
        return 'stable'
    
    # Normalized slope: annual change as proportion of observed range
    normalized_slope = slope / value_range
    
    # Threshold: 10% of range per year is considered meaningful change
    # This is scale-independent (works for 1-4, 1-5, 1-7 scales equally)
    threshold = getattr(Config, 'TREND_SLOPE_THRESHOLD', 0.1)
    
    if abs(normalized_slope) < threshold:
        return 'stable'
    elif normalized_slope > 0:
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


class MirrorKnowledgeGraph:
    """
    Node types:
    - domain: Survey domain/topic (e.g., 'Depression', 'Aggression')
    - question: Survey question
    - option: Valid response option
    
    Edge types:
    - contains: domain -> question (schema)
    - has_option: question -> option (schema)
    - correlates: domain <-> domain (behavioral correlation)
    - similar_to: domain <-> domain (semantic similarity)
    """
    
    def __init__(self):
        if HAS_NETWORKX:
            self.G = nx.DiGraph()
        else:
            self.G = None
        
        # Fallback dict storage
        self.nodes = {'domain': set(), 'question': set(), 'option': set()}
        self.edges = {
            'contains': [],      # domain -> question
            'has_option': [],    # question -> option
            'correlates': [],    # domain <-> domain
            'similar_to': []     # domain <-> domain
        }
        self.node_attrs = {}
        self.edge_attrs = {}
    
    def add_domain_node(self, domain: str, **attrs):
        """Add a domain/topic node"""
        if HAS_NETWORKX:
            self.G.add_node(domain, node_type='domain', **attrs)
        self.nodes['domain'].add(domain)
        self.node_attrs[domain] = {'node_type': 'domain', **attrs}
    
    def add_question_node(self, question: str, domain: str, **attrs):
        """Add a question node and link to domain"""
        if HAS_NETWORKX:
            self.G.add_node(question, node_type='question', domain=domain, **attrs)
            self.G.add_edge(domain, question, relation='contains')
        self.nodes['question'].add(question)
        self.node_attrs[question] = {'node_type': 'question', 'domain': domain, **attrs}
        self.edges['contains'].append((domain, question))
    
    def add_option_node(self, option_id: str, question: str, option_text: str):
        """Add an option node and link to question"""
        node_id = f"{question}::opt_{option_id}"
        if HAS_NETWORKX:
            self.G.add_node(node_id, node_type='option', text=option_text)
            self.G.add_edge(question, node_id, relation='has_option')
        self.nodes['option'].add(node_id)
        self.node_attrs[node_id] = {'node_type': 'option', 'text': option_text}
        self.edges['has_option'].append((question, node_id))
    
    def add_correlation_edge(self, domain1: str, domain2: str, correlation: float,
                              threshold: float = None):
        """Add behavioral correlation edge between domains"""
        if threshold is None:
            threshold = getattr(Config, 'KG_CORRELATION_THRESHOLD', 0.3)
        if abs(correlation) < threshold:
            return
        
        if HAS_NETWORKX:
            self.G.add_edge(domain1, domain2, 
                           relation='correlates', 
                           weight=correlation,
                           direction='positive' if correlation > 0 else 'negative',
                           strength='strong' if abs(correlation) >= 0.5 else 'moderate')
            # Add reverse edge for undirected correlation
            self.G.add_edge(domain2, domain1,
                           relation='correlates',
                           weight=correlation,
                           direction='positive' if correlation > 0 else 'negative',
                           strength='strong' if abs(correlation) >= 0.5 else 'moderate')
        
        self.edges['correlates'].append((domain1, domain2, correlation))
        self.edge_attrs[(domain1, domain2, 'correlates')] = {
            'weight': correlation,
            'direction': 'positive' if correlation > 0 else 'negative',
            'strength': 'strong' if abs(correlation) >= 0.5 else 'moderate'
        }
    
    def add_similarity_edge(self, domain1: str, domain2: str, similarity: float,
                             threshold: float = None):
        """Add semantic similarity edge between domains"""
        if threshold is None:
            threshold = getattr(Config, 'KG_SIMILARITY_THRESHOLD', 0.5)
        if similarity < threshold:
            return
        
        if HAS_NETWORKX:
            self.G.add_edge(domain1, domain2,
                           relation='similar_to',
                           weight=similarity)
        
        self.edges['similar_to'].append((domain1, domain2, similarity))
        self.edge_attrs[(domain1, domain2, 'similar_to')] = {'weight': similarity}
    
    def get_related_domains(self, domain: str, relation_type: str = None, 
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get related domains via correlation or similarity edges.
        
        Both NetworkX and dict fallback now handle bidirectional edges consistently.
        Deduplication is applied to avoid returning the same domain twice.
        """
        related = []
        
        if HAS_NETWORKX and self.G.has_node(domain):
            for _, target, data in self.G.out_edges(domain, data=True):
                if data.get('relation') in ['correlates', 'similar_to']:
                    if relation_type is None or data['relation'] == relation_type:
                        related.append((target, data.get('weight', 0)))
        else:
            # Fallback to dict - handle bidirectional edges for both relation types
            if relation_type is None or relation_type == 'correlates':
                for d1, d2, weight in self.edges['correlates']:
                    if d1 == domain:
                        related.append((d2, weight))
                    elif d2 == domain:
                        related.append((d1, weight))
            
            if relation_type is None or relation_type == 'similar_to':
                for d1, d2, weight in self.edges['similar_to']:
                    if d1 == domain:
                        related.append((d2, weight))
                    elif d2 == domain:
                        related.append((d1, weight))
        
        # Deduplicate: keep highest weight for each domain
        seen = {}
        for target, weight in related:
            if target not in seen or abs(weight) > abs(seen[target]):
                seen[target] = weight
        related = [(t, w) for t, w in seen.items()]
        
        # Sort by absolute weight
        related.sort(key=lambda x: abs(x[1]), reverse=True)
        return related[:top_k]
    
    def get_domain_questions(self, domain: str) -> List[str]:
        """Get all questions under a domain"""
        questions = []
        
        if HAS_NETWORKX and self.G.has_node(domain):
            for _, target, data in self.G.out_edges(domain, data=True):
                if data.get('relation') == 'contains':
                    questions.append(target)
        else:
            for d, q in self.edges['contains']:
                if d == domain:
                    questions.append(q)
        
        return questions
    
    def get_question_options(self, question: str) -> Dict[str, str]:
        """Get valid options for a question"""
        options = {}
        
        if HAS_NETWORKX and self.G.has_node(question):
            for _, target, data in self.G.out_edges(question, data=True):
                if data.get('relation') == 'has_option':
                    opt_data = self.G.nodes[target]
                    opt_id = target.split('::opt_')[-1]
                    options[opt_id] = opt_data.get('text', '')
        else:
            for q, opt_node in self.edges['has_option']:
                if q == question:
                    opt_id = opt_node.split('::opt_')[-1]
                    options[opt_id] = self.node_attrs.get(opt_node, {}).get('text', '')
        
        return options
    
    def get_constraints_for_question(self, question: str) -> Dict[str, Any]:
        """

        Returns valid option range and related domain correlations.
        """
        constraints = {
            'valid_options': {},
            'related_domains': [],
            'correlation_hints': []
        }
        
        # Get valid options (Y_q)
        constraints['valid_options'] = self.get_question_options(question)
        
        # Get domain
        domain = None
        if HAS_NETWORKX:
            if self.G.has_node(question):
                domain = self.G.nodes[question].get('domain')
        else:
            domain = self.node_attrs.get(question, {}).get('domain')
        
        if domain:
            # Get related domains via correlation
            related = self.get_related_domains(domain, top_k=3)
            for rel_domain, corr in related:
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) >= 0.5 else "moderate"
                constraints['related_domains'].append({
                    'domain': rel_domain,
                    'correlation': corr,
                    'direction': direction,
                    'strength': strength
                })
                constraints['correlation_hints'].append(
                    f"{domain} has {strength} {direction} correlation with {rel_domain} (r={corr:.2f})"
                )
        
        return constraints
    
    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics"""
        if HAS_NETWORKX:
            return {
                'total_nodes': self.G.number_of_nodes(),
                'total_edges': self.G.number_of_edges(),
                'domains': len([n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'domain']),
                'questions': len([n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'question']),
                'options': len([n for n, d in self.G.nodes(data=True) if d.get('node_type') == 'option']),
                'correlations': len([1 for u, v, d in self.G.edges(data=True) if d.get('relation') == 'correlates']) // 2,
                'similarities': len([1 for u, v, d in self.G.edges(data=True) if d.get('relation') == 'similar_to'])
            }
        else:
            return {
                'total_nodes': sum(len(v) for v in self.nodes.values()),
                'total_edges': sum(len(v) for v in self.edges.values()),
                'domains': len(self.nodes['domain']),
                'questions': len(self.nodes['question']),
                'options': len(self.nodes['option']),
                'correlations': len(self.edges['correlates']),
                'similarities': len(self.edges['similar_to'])
            }
    
    def to_dict(self) -> Dict:
        """Export graph to dictionary for JSON serialization"""
        return {
            'nodes': {
                'domain': list(self.nodes['domain']),
                'question': list(self.nodes['question']),
                'option': list(self.nodes['option'])
            },
            'edges': {
                'contains': self.edges['contains'],
                'has_option': self.edges['has_option'],
                'correlates': self.edges['correlates'],
                'similar_to': self.edges['similar_to']
            },
            'node_attributes': self.node_attrs,
            'edge_attributes': {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self.edge_attrs.items()},
            'statistics': self.get_stats()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MirrorKnowledgeGraph':
        """Load graph from dictionary"""
        kg = cls()
        
        # Restore nodes
        for domain in data.get('nodes', {}).get('domain', []):
            attrs = data.get('node_attributes', {}).get(domain, {})
            kg.add_domain_node(domain, **{k: v for k, v in attrs.items() if k != 'node_type'})
        
        # Restore edges
        for domain, question in data.get('edges', {}).get('contains', []):
            q_attrs = data.get('node_attributes', {}).get(question, {})
            if question not in kg.nodes['question']:
                kg.add_question_node(question, domain, **{k: v for k, v in q_attrs.items() 
                                                          if k not in ['node_type', 'domain']})
        
        for d1, d2, corr in data.get('edges', {}).get('correlates', []):
            kg.add_correlation_edge(d1, d2, corr)
        
        for d1, d2, sim in data.get('edges', {}).get('similar_to', []):
            kg.add_similarity_edge(d1, d2, sim)
        
        return kg


class KGBuilder:
    """
    Knowledge Graph Builder for MIRROR framework.
    """
    
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
        
        # Initialize Knowledge Graph
        self.kg = MirrorKnowledgeGraph()
        
        # Additional data storage (for backward compatibility)
        self.kg_data = {
            'student_id': student_id,
            'temporal_trends': {},
            'category_profiles': {},
            'category_relationships': {},
            'schema': {},
            'prediction_hints': {},
            'mirror_insights': {},
            'knowledge_graph': {},  # Graph structure
            'metadata': {
                'build_time': None,
                'years_covered': sorted([int(y) for y in self.history.keys()]),
                'has_networkx': HAS_NETWORKX
            }
        }
        
        self._category_yearly_values = defaultdict(lambda: defaultdict(list))
    
    def build_schema_graph(self):
        """
        Build schema structure: Domain -> Question -> Option
        """
        if not os.path.exists(Config.SCHEMA_FILE):
            print(f"[Warning] Schema file not found: {Config.SCHEMA_FILE}")
            return
        
        with open(Config.SCHEMA_FILE, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Add domain nodes with options
        for topic, options in schema_data.items():
            if isinstance(options, list) and options:
                option_type = self._detect_option_type_from_list(options)
                self.kg.add_domain_node(topic, option_type=option_type, option_count=len(options))
                
                # Store options for this domain
                self.kg_data['schema'][topic] = {
                    'options': {str(i+1): opt for i, opt in enumerate(options)},
                    'option_count': len(options),
                    'option_type': option_type
                }
        
        # Link questions to domains
        all_questions = set()
        for year_data in self.history.values():
            all_questions.update(year_data.keys())
        
        for question in all_questions:
            domain = extract_category(question)
            if domain in self.kg.nodes['domain']:
                self.kg.add_question_node(question, domain)
                
                # Add option nodes
                if domain in self.kg_data['schema']:
                    for opt_id, opt_text in self.kg_data['schema'][domain]['options'].items():
                        self.kg.add_option_node(opt_id, question, opt_text)
        
        print(f"[Schema Graph] {len(self.kg.nodes['domain'])} domains, "
              f"{len(self.kg.nodes['question'])} questions")
    
    def build_temporal_trends(self):
        """Build temporal trend data for each question"""
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
                    series.append({'year': int(year), 'value': val})
                    
                    numeric = get_numeric_value(val)
                    if numeric is not None:
                        values.append(numeric)
                        self._category_yearly_values[category][int(year)].append(numeric)
            
            if not series:
                continue
            
            trend_data = {
                'series': series,
                'category': category,
                'last_value': series[-1]['value'] if series else None,
                'topic': category  # For backward compatibility
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
                counter = Counter([s['value'] for s in series])
                mode, count = counter.most_common(1)[0]
                trend_data['mode'] = mode
                trend_data['mode_ratio'] = count / len(series)
            
            # Calculate stability
            unique_values = len(set(s['value'] for s in series))
            if unique_values == 1:
                trend_data['stability'] = 'constant'
            elif unique_values <= 2 and len(series) >= 3:
                trend_data['stability'] = 'highly_stable'
            elif unique_values <= 3:
                trend_data['stability'] = 'stable'
            else:
                trend_data['stability'] = 'variable'
            
            self.kg_data['temporal_trends'][question] = trend_data
    
    def build_correlation_edges(self):
        """
        Build correlation edges between domains.
        
        Note: Correlations are computed from yearly domain means.
        With typical n=3-5 years, statistical significance should be interpreted cautiously.
        Sample size (n_years) is stored in correlation_metadata for transparency.
        
        Threshold justification: |r| >= 0.3 follows Cohen's (1988) convention for
        medium effect sizes, widely used in behavioral/social science research.
        """
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
        correlation_metadata = {}  # Store sample size for statistical transparency
        
        # Get threshold from Config (Issue 7: avoid hardcoded values)
        threshold = getattr(Config, 'KG_CORRELATION_THRESHOLD', 0.3)
        
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
                n_years = len(common_years)
                
                # Add edge to graph using Config threshold
                if abs(corr) >= threshold:
                    self.kg.add_correlation_edge(cat1, cat2, round(corr, 3))
                    correlations[cat1].append((cat2, round(corr, 3)))
                    correlations[cat2].append((cat1, round(corr, 3)))
                    
                    # Store correlation metadata with sample size for transparency
                    correlation_metadata[f"{cat1}|{cat2}"] = {
                        'r': round(corr, 3),
                        'n_years': n_years,
                        'years': common_years
                    }
        
        # Sort by absolute correlation
        for category in correlations:
            correlations[category].sort(key=lambda x: abs(x[1]), reverse=True)
            correlations[category] = correlations[category][:5]
        
        self.kg_data['category_relationships'] = dict(correlations)
        self.kg_data['correlation_metadata'] = correlation_metadata
        
        print(f"[Correlation Edges] {len(self.kg.edges['correlates'])} edges added "
              f"(threshold={threshold}, typical n_years={len(self.history)})")
    
    def build_similarity_edges(self):
        similarity_path = os.path.join(Config.MIRROR_MEMORY_DIR, "category_similarity.json")
        
        if not os.path.exists(similarity_path):
            print(f"[Similarity Edges] category_similarity.json not found, skipping")
            return
        
        try:
            with open(similarity_path, 'r', encoding='utf-8') as f:
                sim_data = json.load(f)
        except Exception as e:
            print(f"[Similarity Edges] Failed to load: {e}")
            return
        
        sim_relationships = sim_data.get('category_relationships', {})
        if not sim_relationships:
            print(f"[Similarity Edges] No relationships found in category_similarity.json")
            return
        
        edge_count = 0
        domains_in_kg = self.kg.nodes['domain']
        
        for cat1, related_list in sim_relationships.items():
            if cat1 not in domains_in_kg:
                continue
            
            for item in related_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    cat2, similarity = item[0], item[1]
                    
                    if cat2 not in domains_in_kg:
                        continue
                    
                    if cat1 == cat2:
                        continue
                    
                    # add_similarity_edge has internal threshold check (>= 0.5)
                    self.kg.add_similarity_edge(cat1, cat2, round(similarity, 3))
                    edge_count += 1
        
        print(f"[Similarity Edges] {len(self.kg.edges['similar_to'])} edges added")
    
    def build_category_profiles(self):
        """Build category-level profiles"""
        category_questions = defaultdict(list)
        
        for question, trend in self.kg_data['temporal_trends'].items():
            category = trend['category']
            category_questions[category].append({
                'question': question,
                'trend': trend.get('trend', 'unknown'),
                'stability': trend.get('stability', 'unknown'),
                'last_value': trend.get('last_value'),
                'data_type': trend.get('data_type', 'unknown')
            })
        
        for category, questions in category_questions.items():
            trends = [q['trend'] for q in questions if q['trend'] != 'unknown']
            stabilities = [q['stability'] for q in questions if q['stability'] != 'unknown']
            
            # Dominant trend
            if trends:
                trend_counts = Counter(trends)
                dominant_trend = trend_counts.most_common(1)[0][0]
            else:
                dominant_trend = 'unknown'
            
            # Stability score
            stability_map = {'constant': 1.0, 'highly_stable': 0.8, 'stable': 0.6, 'variable': 0.3}
            if stabilities:
                stability_score = sum(stability_map.get(s, 0.5) for s in stabilities) / len(stabilities)
            else:
                stability_score = 0.5
            
            # Related categories from KG
            related = self.kg.get_related_domains(category, top_k=3)
            
            # Sudden shift ratio
            shift_count = sum(1 for q in questions if q.get('stability') == 'variable')
            shift_ratio = shift_count / len(questions) if questions else 0
            
            self.kg_data['category_profiles'][category] = {
                'question_count': len(questions),
                'dominant_trend': dominant_trend,
                'stability_score': round(stability_score, 2),
                'sudden_shift_ratio': round(shift_ratio, 2),
                'related_categories': [r[0] for r in related],
                'sample_questions': [q['question'][:50] for q in questions[:3]]
            }
    
    def build_prediction_hints(self):
        """Build prediction hints for each question"""
        for question, trend in self.kg_data['temporal_trends'].items():
            hints = []
            
            series = trend.get('series', [])
            if len(series) >= 3:
                last_values = [s['value'] for s in series[-3:]]
                if len(set(last_values)) == 1:
                    hints.append(f"최근 3년 동일: '{last_values[0]}'")
            
            stability = trend.get('stability', '')
            if stability == 'constant':
                hints.append("완전 일정 패턴")
            elif stability == 'highly_stable':
                hints.append("매우 안정적 패턴")
            
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
                self.kg_data['prediction_hints'][question] = {
                    'hints': hints,
                    'suggested_value': series[-1]['value'] if series else None,
                    'confidence': 'high' if stability in ['constant', 'highly_stable'] else 'medium'
                }
    
    def build_mirror_insights(self):
        """Generate overall insights using LLM"""
        category_summaries = []
        for cat, profile in list(self.kg_data['category_profiles'].items())[:5]:
            trend = profile.get('dominant_trend', 'unknown')
            stability = profile.get('stability_score', 0)
            count = profile.get('question_count', 0)
            related = profile.get('related_categories', [])
            related_str = f", related: {','.join(related[:2])}" if related else ""
            category_summaries.append(f"- {cat}: {count} items, trend={trend}, stability={stability:.2f}{related_str}")
        
        # Get strong correlations from KG
        strong_correlations = []
        for d1, d2, corr in self.kg.edges['correlates']:
            if abs(corr) >= 0.5:
                direction = "positive" if corr > 0 else "negative"
                strong_correlations.append(f"- {d1} <-> {d2}: {direction} correlation ({corr})")
        
        prompt = f"""Analyze student {self.student_id}'s survey data.

[Category Summary]
{chr(10).join(category_summaries[:10])}

[Strong Correlations]
{chr(10).join(strong_correlations[:5]) if strong_correlations else 'None'}

[KG Statistics]
{self.kg.get_stats()}

Summarize in 3-4 sentences:
1. Overall developmental characteristics
2. Key inter-category patterns
3. Considerations for 2023 predictions"""
        
        try:
            response = self.llm.invoke(prompt)
            self.kg_data['mirror_insights'] = {
                'raw_analysis': response.content.strip(),
                'strong_correlation_count': len(strong_correlations)
            }
        except Exception as e:
            self.kg_data['mirror_insights'] = {
                'raw_analysis': f"Analysis failed: {str(e)[:50]}",
                'strong_correlation_count': 0
            }
    
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
        """Build complete knowledge graph"""
        if not self.history:
            return self.kg_data
        
        # 1. Build schema graph (Domain -> Question -> Option)
        self.build_schema_graph()
        
        # 2. Build temporal trends
        self.build_temporal_trends()
        
        # 3. Build correlation edges (Pearson correlation)
        self.build_correlation_edges()
        
        # 4. Build similarity edges (Semantic similarity)
        self.build_similarity_edges()
        
        # 5. Build category profiles
        self.build_category_profiles()
        
        # 6. Build prediction hints
        self.build_prediction_hints()
        
        # 7. Build insights
        self.build_mirror_insights()
        
        # Store graph structure
        self.kg_data['knowledge_graph'] = self.kg.to_dict()
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
    
    stats = builder.kg.get_stats()
    stats.update({
        'trends': len(builder.kg_data['temporal_trends']),
        'profiles': len(builder.kg_data['category_profiles']),
        'hints': len(builder.kg_data['prediction_hints'])
    })
    
    return output_path, elapsed, stats


def build_kg_batch(student_ids: List[str], output_dir: str = None,
                   max_workers: int = 2, skip_existing: bool = False):
    import time
    
    print(f"Building KG for {len(student_ids)} students...")
    print(f"Workers: {max_workers}, Skip existing: {skip_existing}")
    print(f"NetworkX available: {HAS_NETWORKX}")
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
                        'corr': stats.get('correlations', 0)
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
            avg_corr = sum(r[3].get('correlations', 0) for r in success_results) / len(success_results)
            print(f"Avg correlations per student: {avg_corr:.1f}")
    
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
        print("MIRROR Knowledge Graph Builder")
        print("=" * 40)
        print("Usage:")
        print("  python build_kg.py --student 10001")
        print("  python build_kg.py --all --workers 2")
        print("  python build_kg.py --all --rebuild")
