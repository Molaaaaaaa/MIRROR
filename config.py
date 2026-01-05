"""
설정 파일 (v6 데이터셋 기반)
"""
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SCHEMA_FILE = os.path.join(BASE_DIR, "data", "schema.json")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_cache")
    AGENT_MEMORY_DIR = os.path.join(BASE_DIR, "agent_memory")
    
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = "qwen3:14b-q8_0"
    
    BATCH_SIZE = 16
    CONCURRENT_REQUESTS = 16
    LLM_TIMEOUT = 600
    NUM_CTX = 4096
    
    KNOWLEDGE_YEARS = [2018, 2019, 2020, 2021, 2022]
    ANALYSIS_YEARS = [2019, 2020, 2021, 2022]
    TARGET_YEAR = 2023
    
    NEO4J_URL = os.getenv("NEO4J_URL", "")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    INPUT_VARIABLES = [
        "시/도(학교 기준)",
        "도시규모(학교 기준)",
        "시/도(거주지 기준)",
        "생년",
        "성별",
        "형제자매 수(본인포함)"
    ]
    
    TARGET_CATEGORIES = ["공격성", "학교 폭력"]
    
    # 과거 데이터에서 제외할 카테고리 (--exclude-target 옵션 사용 시)
    EXCLUDED_CATEGORIES = ["공격성", "학교 폭력", "현실비행 경험 유무 및 빈도"]
    
    # 부분 제외 (--exclude-partial 옵션 사용 시, 공격성은 포함)
    EXCLUDED_CATEGORIES_PARTIAL = ["학교 폭력", "현실비행 경험 유무 및 빈도"]
    
    TARGET_DELINQUENCY_ITEMS = [
        "술 마시기",
        "심한 욕설과 폭언",
        "무단결석",
        "다른 사람 심하게 놀리거나 조롱하기"
    ]
    
    # [DEPRECATED] AI Agent가 동적으로 RAG/KG/LLM을 통해 관련 카테고리를 탐색하므로 사용하지 않음
    # RELATED_CATEGORIES = {
    #     "학교 폭력": ["공격성", "사회적 위축", "친구관계", "협동심"],
    #     ...
    # }
    
    # 컨텍스트 길이 설정 (할루시네이션 방지와 정보량 균형)
    NARRATIVE_MAX_LENGTH = 400      # 성장 서사 최대 길이
    KG_INSIGHT_MAX_LENGTH = 300     # KG AI 인사이트 최대 길이
    YEARLY_SUMMARY_MAX_LENGTH = 300 # 연간 변화 요약 최대 길이
    
    # 부정적 행동 관련 카테고리 키워드 (스키마에서 필터링용)
    NEGATIVE_BEHAVIOR_KEYWORDS = ["공격성", "위축", "우울", "주의", "신체", "무기력"]
    
    # 관련 카테고리 우선순위 (예측 시 참조할 카테고리)
    RELATED_CATEGORY_PRIORITY = ["공격성", "사회적 위축", "우울", "주의집중", "친구관계"]
    
    # LTM/KG 구축 설정
    LTM_WORKERS = 2
    KG_WORKERS = 2