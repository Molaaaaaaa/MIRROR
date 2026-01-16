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
    
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = "qwen3:14b-q8_0"
    
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