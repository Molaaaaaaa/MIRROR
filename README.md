# MIRROR: Multi-view Inference via Retrospective Retrieval for Ontological Representation of Persona

## Overview

MIRROR is a model-agnostic framework for closed-set next-year prediction from longitudinal survey data. It performs single-pass, item-conditioned inference by:

1. **Retrospective Evidence Retrieval (RER)**: Caching historical evidence as question-answer-year documents
2. **Longitudinal Trend Extraction (LTE)**: Summarizing multi-year responses into static and dynamic persona profiles
3. **Knowledge Graph with Consistency Constraints (KG)**: Encoding topic-item-option relations and behavioral correlations

## Project Structure
```
MIRROR/
├── mirror_framework.py              # LLM reasoning by MIRROR framework
├── build_ltm.py                     # Longitudinal Trend Extraction (LTE)
├── build_kg.py                      # Per-student Knowledge Graph
├── build_global_kg.py               # Global Knowledge Graph
├── build_behavioral_correlation.py  # KG - Behavioral correlation edges
├── build_category_similarity.py     # KG - Semantic similarity edges
├── tools.py                         # Framework tools (RAG, STM, LTM, KG)
├── memory_manager.py                # LTM/STM management
├── run_experiments.py               # Experiment runner
├── config.py                        # Configuration settings
├── data_constants.py                # Dataset-specific constants
├── utils.py                         # Utility functions
├── requirements.txt                 # Dependencies
├── mirror_memory/                   # Generated LTM/KG cache (auto-created)
├── chroma_cache/                    # RAG vector store (auto-created)
├── results/                         # Experiment results (auto-created)
└── data/
    ├── students/                    # Student longitudinal survey data
    └── schema.json                  # Question-option mapping schema
```

## Requirements

- Python 3.8+
- Ollama (for local LLM inference)
- NVIDIA GPU (recommended)

## Installation
```bash
# Clone repository
git clone https://github.com/anonymous/MIRROR.git
cd MIRROR

# Install dependencies
pip install -r requirements.txt

# Pull LLM model
ollama pull qwen3:14b-q8_0
```

## Quick Start

### 1. Build Knowledge Repository (Offline Stage)
```bash
# Build Long-Term Memory (Persona Profiles)
python build_ltm.py --all

# Build Knowledge Graph
python build_kg.py --all

# Build Global Behavioral Correlations
python build_behavioral_correlation.py

# Build Category Semantic Similarity
python build_category_similarity.py
```

### 2. Run Prediction Experiments (Online Stage)
```bash
# Single student experiment
python run_experiments.py --student <student_id> --methods MIRROR

# All students with full MIRROR framework
python run_experiments.py --all --methods MIRROR

# Compare with baselines
python run_experiments.py --all --methods 2018_only 2022_only LLM_only RER RER_LTE MIRROR

# Experimental settings (S2: Violence-blinded)
python run_experiments.py --all --methods MIRROR --exclude-target

# Experimental settings (S3: S2 + Aggression)
python run_experiments.py --all --methods MIRROR --exclude-partial
```

## Configuration

Key settings in `config.py`:
```python
OLLAMA_MODEL = "qwen3:14b-q8_0"  # Backbone LLM
TARGET_YEAR = 2023               # Prediction target year
KNOWLEDGE_YEARS = [2018, 2019, 2020, 2021, 2022]  # Training years
```

## Experimental Settings

| Setting | Description |
|---------|-------------|
| S1 (Full-history) | Use all 2018-2022 survey responses |
| S2 (Violence-blinded) | Exclude Aggression, Delinquency, School Violence |
| S3 (S2 + Aggression) | S2 with Aggression history restored |