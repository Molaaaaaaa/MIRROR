"""
유틸리티 함수 (Final Version - Fuzzy Matching + Improved Fallback)
파일명: utils.py
"""
import os
import glob
import ast
import pandas as pd
import re
import json
import traceback
from typing import List, Dict, Tuple, Optional, Any
from config import Config

# Config에서 카테고리 설정 가져오기 (하드코딩 제거)
def _get_excluded_categories():
    return getattr(Config, 'EXCLUDED_CATEGORIES', [])

def _get_excluded_categories_partial():
    return getattr(Config, 'EXCLUDED_CATEGORIES_PARTIAL', [])


def normalize_question_text(question: str) -> str:
    """질문 텍스트 정규화"""
    if not question:
        return ""
    text = re.sub(r'\s+', ' ', question.strip())
    text = text.replace('，', ',').replace('．', '.').replace('"', '"').replace('"', '"')
    return text


def get_all_student_ids(data_dir: str) -> List[str]:
    """데이터 디렉토리에서 학생 ID 목록 추출"""
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
    """질문 텍스트에서 카테고리 추출 (대괄호 기준)"""
    match = re.match(r'\[([^\]]+)\]', question)
    if match:
        return match.group(1)
    return "기타"


def load_student_data(data_dir: str, student_id: str, exclude_target: bool = False, exclude_partial: bool = False) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """학생 데이터 로드
    
    Args:
        exclude_target: True면 EXCLUDED_CATEGORIES 전체(공격성, 학교폭력, 현실비행) 제외
        exclude_partial: True면 EXCLUDED_CATEGORIES_PARTIAL(학교폭력, 현실비행)만 제외 (공격성은 포함)
    """
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

            if "설문 문항" in cols: q_col = "설문 문항"
            elif "문항" in cols: q_col = "문항"
            
            if "응답 내용" in cols: a_col = "응답 내용"
            elif "답변" in cols: a_col = "답변"
            elif "응답" in cols: a_col = "응답"
            else:
                cand = [c for c in cols if "답" in c and not any(x in c for x in ["코드", "자", "년도"])]
                if cand: a_col = cand[0]

            if q_col and a_col:
                df_clean = df.dropna(subset=[q_col, a_col])
                valid_data = {}
                
                for _, row in df_clean.iterrows():
                    q_text = normalize_question_text(str(row[q_col]))
                    a_text = str(row[a_col]).strip()
                    
                    if not input_vars:
                        if "성별" in q_text: input_vars["성별"] = a_text
                        elif "생년" in q_text: input_vars["생년"] = a_text
                        elif "거주지" in q_text or "시/도" in q_text: input_vars["거주지"] = a_text

                    if '[' in q_text and ']' in q_text:
                        category = extract_category(q_text)
                        
                        if exclude_target:
                            if category in _get_excluded_categories():
                                continue
                        
                        if exclude_partial:
                            if category in _get_excluded_categories_partial():
                                continue
                        
                        if "중1" not in a_text and "패널" not in a_text and len(a_text) < 100:
                            valid_data[q_text] = a_text
                
                if valid_data:
                    history[year] = valid_data
        except Exception: 
            pass

    if not input_vars: input_vars = {"info": "Unknown"}
    return input_vars, history


load_student_all_data = load_student_data


def load_ground_truth(data_dir: str, student_id: str, target_year: int = 2023) -> Dict[str, str]:
    """정답 데이터 로드"""
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
        
        q_col = "설문 문항" if "설문 문항" in cols else None
        a_col = "응답 내용" if "응답 내용" in cols else None
        
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
    """타겟 문항 필터링
    
    Args:
        all_targets: 전체 문항 리스트
        partial_only: True면 학교폭력, 현실비행만 필터링 (공격성 제외)
    """
    filtered = []
    target_cats = Config.TARGET_CATEGORIES
    delinquency_items = Config.TARGET_DELINQUENCY_ITEMS
    
    if partial_only:
        target_cats = ["학교 폭력"]

    for t in all_targets:
        q_text = t['question']
        category = t['category']
        
        if category in target_cats:
            filtered.append(t)
            continue
            
        if "현실비행" in category or "현실비행" in q_text:
            for item in delinquency_items:
                if item in q_text:
                    filtered.append(t)
                    break
    return filtered


def generate_targets_from_student_data(student_id: str, option_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    [개선] 현재 학생의 데이터에서 직접 타겟 질문 생성
    """
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
            options = {"1": "그렇다", "2": "아니다"}
            
        final_targets.append({
            "question": t['question'],
            "category": cat,
            "options": options
        })
        
    return final_targets


def generate_targets_from_options(option_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    [Auto Fix] JSON 파일이 옵션 정보만 담고 있을 때,
    실제 데이터에서 질문을 찾아 타겟 리스트를 자동 생성
    """
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
            options = {"1": "그렇다", "2": "아니다"}
            
        final_targets.append({
            "question": t['question'],
            "category": cat,
            "options": options
        })
        
    print(f"  -> Generated {len(final_targets)} targets from data.")
    return final_targets


def get_target_questions(filepath: str) -> List[Dict[str, Any]]:
    """예측 대상 문항 로드 (자동 복구 기능 포함)"""
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
                        q = item.get('question') or item.get('설문 문항', '')
                        q = normalize_question_text(q)
                        opts = item.get('options') or item.get('응답 내용', {})
                        targets.append({
                            "question": q,
                            "options": opts if isinstance(opts, dict) else {"1": "그렇다", "2": "아니다"},
                            "category": item.get('category', extract_category(q))
                        })

        else:
            try: df = pd.read_csv(filepath, encoding='utf-8-sig')
            except: df = pd.read_csv(filepath, encoding='cp949')
            
            q_col = next((c for c in df.columns if '설문 문항' in c), df.columns[0])
            opt_col = next((c for c in df.columns if '응답 내용' in c), df.columns[1])

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
                if not options: options = {"1": "그렇다", "2": "아니다"}
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
    """LLM 출력에서 답변 번호 추출"""
    if not text: return "0"
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    if text.isdigit(): return text
    match = re.search(r'(?:Option|답|번)?\s*(\d+)', text)
    if match: return match.group(1)
    return "0"


def get_fallback_prediction(temporal_data: List[Dict], options: Dict[str, str]) -> str:
    """개선된 Fallback 예측 로직
    
    우선순위:
    1. 최근 값과 옵션 매칭
    2. 최빈값 (Mode)
    3. 중앙값에 가장 가까운 옵션
    4. 첫 번째 옵션 (최후의 수단)
    """
    if not options:
        return "0"
    
    option_keys = list(options.keys())
    option_values = list(options.values())
    
    if temporal_data:
        # 1. 최근 값과 옵션 매칭 시도
        last_value = temporal_data[-1].get('value', '')
        for k, v in options.items():
            # 정확한 매칭
            if v == last_value:
                return k
            # 부분 매칭 (값이 옵션에 포함되거나 그 반대)
            if last_value and (last_value in v or v in last_value):
                return k
        
        # 2. 최빈값 계산
        values = [p.get('value', '') for p in temporal_data]
        if values:
            from collections import Counter
            value_counts = Counter(values)
            most_common_value = value_counts.most_common(1)[0][0]
            
            for k, v in options.items():
                if v == most_common_value or most_common_value in v or v in most_common_value:
                    return k
        
        # 3. 숫자형 값이면 추세 기반 예측
        try:
            numeric_values = []
            for p in temporal_data:
                val = p.get('value', '')
                # 옵션 값을 숫자로 변환 시도
                for k, v in options.items():
                    if v == val:
                        numeric_values.append(int(k))
                        break
            
            if len(numeric_values) >= 2:
                # 추세 계산 (단순 평균 변화량)
                changes = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
                avg_change = sum(changes) / len(changes)
                predicted = numeric_values[-1] + avg_change
                
                # 가장 가까운 옵션 선택
                closest_key = min(option_keys, key=lambda k: abs(int(k) - predicted))
                return closest_key
        except (ValueError, TypeError):
            pass
    
    # 4. 중앙 옵션 반환 (첫 번째 옵션보다 나은 기본값)
    mid_idx = len(option_keys) // 2
    return option_keys[mid_idx] if option_keys else "0"


def format_temporal_trace(patterns: List[Dict]) -> str:
    """시계열 데이터 포맷팅"""
    if not patterns: return ""
    return " -> ".join([f"{p['year']}: {p['value']}" for p in patterns])


def format_changes_summary(changes: Dict) -> str:
    """변화 요약 포맷팅"""
    if not changes: return ""
    return str(changes)
