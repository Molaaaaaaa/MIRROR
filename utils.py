import os
import glob
import ast
import pandas as pd
import re
import json
from typing import List, Dict, Tuple, Any
from config import Config
from data_constants import (
    CSV_COLUMNS,
    INPUT_VARIABLE_KEYWORDS,
    DEFAULT_OPTIONS,
    TARGET_CATEGORIES,
    EXCLUDED_QUESTIONS,
)

def _get_excluded_categories():
    return getattr(Config, 'EXCLUDED_CATEGORIES', [])

def _get_excluded_categories_partial():
    return getattr(Config, 'EXCLUDED_CATEGORIES_PARTIAL', [])


def normalize_question_text(question: str) -> str:
    if not question:
        return ""
    text = re.sub(r'\s+', ' ', question.strip())
    text = text.replace('，', ',').replace('．', '.').replace('"', '"').replace('"', '"')
    return text


def get_all_student_ids(data_dir: str) -> List[str]:
    student_ids = set()
    if not os.path.exists(data_dir):
        return []

    try:
        items = os.listdir(data_dir)
        for item in items:
            if item.isdigit() and os.path.isdir(os.path.join(data_dir, item)):
                student_ids.add(item)
    except OSError:
        pass
            
    csv_files = glob.glob(os.path.join(data_dir, "*_*.csv"))
    for f in csv_files:
        filename = os.path.basename(f)
        match = re.match(r"(\d+)_", filename)
        if match:
            student_ids.add(match.group(1))
            
    return sorted(list(student_ids))


def extract_category(question: str) -> str:
    match = re.match(r'\[([^\]]+)\]', question)
    if match:
        return match.group(1)
    return "기타"


def load_student_data(data_dir: str, student_id: str, exclude_target: bool = False, exclude_partial: bool = False) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    input_vars = {}
    history = {}
    
    student_dir = os.path.join(data_dir, str(student_id))
    if not os.path.exists(student_dir):
        if os.path.exists(os.path.join(data_dir, f"{student_id}_2018.csv")):
            student_dir = data_dir
        else:
            return input_vars, history

    if student_dir == data_dir:
        csv_files = glob.glob(os.path.join(student_dir, f"{student_id}_*.csv"))
    else:
        csv_files = glob.glob(os.path.join(student_dir, "*.csv"))

    for fpath in csv_files:
        try:
            filename = os.path.basename(fpath)
            year_matches = re.findall(r'(20\d{2})', filename)
            year = year_matches[-1] if year_matches else "Unknown"

            try: df = pd.read_csv(fpath, encoding='utf-8-sig')
            except: df = pd.read_csv(fpath, encoding='cp949')

            df.columns = [str(c).strip() for c in df.columns]
            cols = df.columns.tolist()

            q_col = None
            a_col = None

            # Find question column
            for col_name in CSV_COLUMNS['question']:
                if col_name in cols:
                    q_col = col_name
                    break
            
            # Find answer column
            for col_name in CSV_COLUMNS['answer']:
                if col_name in cols:
                    a_col = col_name
                    break
            
            if a_col is None:
                # Fallback: find column containing answer keyword but excluding filter words
                answer_kw = CSV_COLUMNS['answer_keyword']
                cand = [c for c in cols if answer_kw in c and not any(x in c for x in CSV_COLUMNS['answer_filter_exclude'])]
                if cand: a_col = cand[0]

            if q_col and a_col:
                df_clean = df.dropna(subset=[q_col, a_col])
                valid_data = {}
                
                for _, row in df_clean.iterrows():
                    q_text = normalize_question_text(str(row[q_col]))
                    a_text = str(row[a_col]).strip()
                    
                    if not input_vars:
                        # Extract demographic info using keywords
                        gender_kw = INPUT_VARIABLE_KEYWORDS['gender']
                        birth_kw = INPUT_VARIABLE_KEYWORDS['birth_year']
                        region_kws = INPUT_VARIABLE_KEYWORDS['region']
                        
                        if gender_kw in q_text: input_vars["gender"] = a_text
                        elif birth_kw in q_text: input_vars["birth_year"] = a_text
                        elif any(kw in q_text for kw in region_kws): input_vars["region"] = a_text

                    if '[' in q_text and ']' in q_text:
                        category = extract_category(q_text)
                        
                        if exclude_target:
                            if category in _get_excluded_categories():
                                continue
                        
                        if exclude_partial:
                            if category in _get_excluded_categories_partial():
                                continue
                        
                        # Filter invalid responses
                        invalid_keywords = CSV_COLUMNS['invalid_response']
                        if not any(kw in a_text for kw in invalid_keywords) and len(a_text) < 100:
                            valid_data[q_text] = a_text
                
                if valid_data:
                    history[year] = valid_data
        except Exception: 
            pass

    if not input_vars: input_vars = {"info": "Unknown"}
    return input_vars, history


load_student_all_data = load_student_data


def load_ground_truth(data_dir: str, student_id: str, target_year: int = 2023) -> Dict[str, str]:
    ground_truth = {}
    path1 = os.path.join(data_dir, str(student_id), f"{student_id}_{target_year}.csv")
    path2 = os.path.join(data_dir, f"{student_id}_{target_year}.csv")
    
    file_path = path1 if os.path.exists(path1) else (path2 if os.path.exists(path2) else None)
    if not file_path: return ground_truth

    try:
        try: df = pd.read_csv(file_path, encoding='utf-8-sig')
        except: df = pd.read_csv(file_path, encoding='cp949')
        
        df.columns = [str(c).strip() for c in df.columns]
        cols = df.columns.tolist()
        
        # Find columns using constants
        q_col = CSV_COLUMNS['question'][0] if CSV_COLUMNS['question'][0] in cols else None
        a_col = CSV_COLUMNS['answer'][0] if CSV_COLUMNS['answer'][0] in cols else None
        
        if q_col and a_col:
            df = df.dropna(subset=[q_col, a_col])
            for _, row in df.iterrows():
                q_text = normalize_question_text(str(row[q_col]))
                a_text = str(row[a_col]).strip()
                if '[' in q_text and ']' in q_text:
                    ground_truth[q_text] = a_text
    except: pass
    return ground_truth


def filter_target_questions(all_targets: List[Dict], partial_only: bool = False) -> List[Dict]:
    """Filter target questions - INCLUDE only target categories (25 items)."""
    filtered = []
    target_cats = TARGET_CATEGORIES
    
    for t in all_targets:
        category = t['category']
        question = t['question']
        
        # Include ONLY if category is in target list
        if category not in target_cats:
            continue
        
        # Exclude specific questions (e.g., smoking)
        if any(excl in question for excl in EXCLUDED_QUESTIONS):
            continue
        
        # Include target category questions
        filtered.append(t)
    
    return filtered


def filter_non_target_questions(all_targets: List[Dict]) -> List[Dict]:
    """
    Filter target questions - EXCLUDE target categories (25 items).
    Returns the remaining ~187 items for prediction.
    
    This is the inverse of filter_target_questions().
    Used for predicting non-violence related survey items.
    """
    filtered = []
    target_cats = TARGET_CATEGORIES  # [공격성, 학교 폭력, 현실비행]
    
    for t in all_targets:
        category = t['category']
        
        # EXCLUDE if category is in target list (inverse of filter_target_questions)
        if category in target_cats:
            continue
        
        # Include all other categories
        filtered.append(t)
    
    return filtered


def generate_targets_from_student_data(student_id: str, option_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    _, history = load_student_data(Config.DATA_DIR, student_id)
    
    unique_questions = set()
    for year_data in history.values():
        for q in year_data.keys():
            unique_questions.add(q)
            
    temp_targets = []
    for q in unique_questions:
        cat = extract_category(q)
        temp_targets.append({
            "question": q,
            "category": cat,
            "options": {}
        })
        
    filtered = filter_target_questions(temp_targets)
    
    final_targets = []
    for t in filtered:
        cat = t['category']
        opts_list = option_map.get(cat)
        
        if not opts_list:
            for map_cat, map_opts in option_map.items():
                if map_cat in cat or cat in map_cat:
                    opts_list = map_opts
                    break
        
        if opts_list and isinstance(opts_list, list):
            options = {str(i+1): v for i, v in enumerate(opts_list)}
        else:
            options = DEFAULT_OPTIONS.copy()
            
        final_targets.append({
            "question": t['question'],
            "category": cat,
            "options": options
        })
        
    return final_targets


def generate_targets_from_options(option_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    print("  [Auto Fix] Generating target questions from raw data using provided option schema...")
    
    student_ids = get_all_student_ids(Config.DATA_DIR)
    if not student_ids:
        print("  [Error] No student data found to generate targets.")
        return []
    
    _, history = load_student_data(Config.DATA_DIR, student_ids[0])
    
    unique_questions = set()
    for year_data in history.values():
        for q in year_data.keys():
            unique_questions.add(q)
            
    temp_targets = []
    for q in unique_questions:
        cat = extract_category(q)
        temp_targets.append({
            "question": q,
            "category": cat,
            "options": {}
        })
        
    filtered = filter_target_questions(temp_targets)
    
    final_targets = []
    for t in filtered:
        cat = t['category']
        opts_list = option_map.get(cat)
        
        if not opts_list:
            for map_cat, map_opts in option_map.items():
                if map_cat in cat or cat in map_cat:
                    opts_list = map_opts
                    break
        
        if opts_list and isinstance(opts_list, list):
            options = {str(i+1): v for i, v in enumerate(opts_list)}
        else:
            options = DEFAULT_OPTIONS.copy()
            
        final_targets.append({
            "question": t['question'],
            "category": cat,
            "options": options
        })
        
    print(f"  -> Generated {len(final_targets)} targets from data.")
    return final_targets


def get_target_questions(filepath: str) -> List[Dict[str, Any]]:
    targets = []
    
    if not os.path.exists(filepath):
        print(f"[Error] Target file not found: {filepath}")
        return targets

    try:
        if filepath.lower().endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                first_val = next(iter(data.values())) if data else None
                if isinstance(first_val, list) and first_val and isinstance(first_val[0], str):
                    return generate_targets_from_options(data)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Get question from JSON (Korean key: 설문 문항)
                        q = item.get('question') or item.get(CSV_COLUMNS['question'][0], '')
                        q = normalize_question_text(q)
                        # Get options from JSON (Korean key: 응답 내용)
                        opts = item.get('options') or item.get(CSV_COLUMNS['answer'][0], {})
                        targets.append({
                            "question": q,
                            "options": opts if isinstance(opts, dict) else DEFAULT_OPTIONS.copy(),
                            "category": item.get('category', extract_category(q))
                        })

        else:
            try: df = pd.read_csv(filepath, encoding='utf-8-sig')
            except: df = pd.read_csv(filepath, encoding='cp949')
            
            # Find columns using Korean keywords from constants
            q_col = next((c for c in df.columns if CSV_COLUMNS['question'][0] in c), df.columns[0])
            opt_col = next((c for c in df.columns if CSV_COLUMNS['answer'][0] in c), df.columns[1])

            for _, row in df.iterrows():
                q = normalize_question_text(str(row[q_col]))
                opts = str(row[opt_col]).strip()
                options = {}
                if opts and opts != 'nan':
                    try:
                        if opts.startswith('['):
                            lst = ast.literal_eval(opts)
                            for i, v in enumerate(lst): options[str(i+1)] = v
                        else: options = {"1": opts}
                    except: options = {"1": opts}
                if not options: options = DEFAULT_OPTIONS.copy()
                targets.append({
                    "question": q,
                    "options": options,
                    "category": extract_category(q)
                })

    except Exception as e:
        print(f"[Error] Failed to load target questions: {e}")
        
    return targets


load_target_questions = get_target_questions


def clean_llm_output(text: str) -> str:
    """Clean LLM output to extract answer number"""
    if not text: return "0"
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    # Remove student IDs (4-5 digit numbers) to prevent misparse
    text = re.sub(r'\b\d{4,5}\b', '', text)
    # Pure digit
    if text.strip().isdigit() and len(text.strip()) == 1: return text.strip()
    # Priority 1: "답: N" or "Answer: N" (highest confidence)
    match = re.search(r'답[:\s]*([1-6])', text)
    if match: return match.group(1)
    match = re.search(r'[Aa]nswer[:\s]*([1-6])', text)
    if match: return match.group(1)
    # Priority 2: English reasoning patterns
    for pattern in [
        r'(?:the\s+)?answer\s+(?:is|would be|should be)[:\s]*([1-6])',
        r'(?:I\s+)?predict[:\s]*([1-6])',
        r'(?:I\s+)?(?:would\s+)?choose[:\s]*([1-6])',
        r'option[:\s]*([1-6])',
        r'select[:\s]*([1-6])',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return match.group(1)
    # Priority 3: Last 100 chars standalone digit (likely conclusion)
    last_part = text[-100:] if len(text) > 100 else text
    match = re.search(r'\b([1-6])\b(?![\d])', last_part)
    if match: return match.group(1)
    # Priority 4: First standalone 1-6 digit
    match = re.search(r'\b([1-6])\b', text)
    if match: return match.group(1)
    return "0"


def get_fallback_prediction(temporal_data: List[Dict], options: Dict[str, str]) -> str:
    if not options:
        return "0"
    
    option_keys = list(options.keys())
    option_values = list(options.values())
    
    if temporal_data:
        last_value = temporal_data[-1].get('value', '')
        for k, v in options.items():
            if v == last_value:
                return k
            if last_value and (last_value in v or v in last_value):
                return k
        
        values = [p.get('value', '') for p in temporal_data]
        if values:
            from collections import Counter
            value_counts = Counter(values)
            most_common_value = value_counts.most_common(1)[0][0]
            
            for k, v in options.items():
                if v == most_common_value or most_common_value in v or v in most_common_value:
                    return k
        
        try:
            numeric_values = []
            for p in temporal_data:
                val = p.get('value', '')
                for k, v in options.items():
                    if v == val:
                        numeric_values.append(int(k))
                        break
            
            if len(numeric_values) >= 2:
                changes = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
                avg_change = sum(changes) / len(changes)
                predicted = numeric_values[-1] + avg_change
                
                closest_key = min(option_keys, key=lambda k: abs(int(k) - predicted))
                return closest_key
        except (ValueError, TypeError):
            pass
    
    mid_idx = len(option_keys) // 2
    return option_keys[mid_idx] if option_keys else "0"


def format_temporal_trace(patterns: List[Dict]) -> str:
    if not patterns: return ""
    return " -> ".join([f"{p['year']}: {p['value']}" for p in patterns])


def format_changes_summary(changes: Dict) -> str:
    if not changes: return ""
    return str(changes)
