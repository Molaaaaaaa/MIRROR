"""
Configuration file for MIRROR Framework
"""
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "students")
    SCHEMA_FILE = os.path.join(BASE_DIR, "data", "schema.json")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_cache")
    MIRROR_MEMORY_DIR = os.path.join(BASE_DIR, "mirror_memory")

    # LLM Provider: "ollama" (local), "openai" (GPT), "deepseek" (DeepSeek API)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama Configuration (local models)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b-q8_0")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b-it-q8_0")
    # OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3-27b-q8:latest")

    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    # DeepSeek API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    BATCH_SIZE = 16
    CONCURRENT_REQUESTS = 16
    LLM_TIMEOUT = 600
    NUM_CTX = 4096
    
    KNOWLEDGE_YEARS = [2018, 2019, 2020, 2021, 2022]
    ANALYSIS_YEARS = [2019, 2020, 2021, 2022]
    TARGET_YEAR = 2023
    
    # Context length settings
    NARRATIVE_MAX_LENGTH = 300       # Max length for growth narrative
    NARRATIVE_SHORT_LENGTH = 200     # Shorter narrative for existing questions
    KG_INSIGHT_MAX_LENGTH = 300      # Max length for KG AI insights
    YEARLY_SUMMARY_MAX_LENGTH = 300  # Max length for yearly change summary
    
    # KG threshold settings
    KG_CORRELATION_THRESHOLD = 0.3   # Min |r| for correlation edges (Cohen's medium effect)
    KG_SIMILARITY_THRESHOLD = 0.5    # Min similarity for semantic edges
    
    # Trend detection settings
    TREND_SLOPE_THRESHOLD = 0.1      # Normalized slope threshold (10% of range per year)
    
    # Cache settings
    KG_CACHE_MAX_SIZE = 50           # Max number of KG objects in memory
    
    # LTM/KG build settings
    LTM_WORKERS = 2
    KG_WORKERS = 2
    
    # Import data-dependent constants from data_constants.py
    # These are separated for clarity and dataset portability
    from data_constants import (
        INPUT_VARIABLE_NAMES,
        TARGET_CATEGORIES,
        EXCLUDED_CATEGORIES,
        EXCLUDED_CATEGORIES_PARTIAL,
        TARGET_DELINQUENCY_ITEMS,
        NEGATIVE_BEHAVIOR_KEYWORDS,
        RELATED_CATEGORY_PRIORITY,
    )
    
    INPUT_VARIABLES = list(INPUT_VARIABLE_NAMES.values())