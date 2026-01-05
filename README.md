# MIRROR: Multi-view Inference via Retrospective Retrieval for Ontological Representation of Persona

Official implementation of the paper "MIRROR: Multi-view Inference via Retrospective Retrieval for Ontological Representation of Persona" (Anonymous Submission).

## Overview

MIRROR is a model-agnostic framework for closed-set next-year prediction from longitudinal survey data. It performs single-pass, item-conditioned inference by:

1. **Retrospective Evidence Retrieval (RER)**: Caching historical evidence as question-answer-year documents
2. **Longitudinal Trend Extraction (LTE)**: Summarizing multi-year responses into static and dynamic persona profiles
3. **Knowledge Graph with Consistency Constraints (KG)**: Encoding topic-item-option relations and behavioral correlations

## Project Structure
```
MIRROR/
├── agent.py                         # LLM reasoning agent (Section 2.4)
├── build_ltm.py                     # Longitudinal Trend Extraction (Section 2.2)
├── build_kg.py                      # Knowledge Graph construction (Section 2.3)
├── build_behavioral_correlation.py  # KG - Behavioral correlation edges
├── build_category_similarity.py     # KG - Semantic similarity edges
├── tools.py                         # Agent tools including RER (Section 2.1)
├── memory_manager.py                # LTM/STM management
├── run_experiments.py               # Experiment runner (Section 4)
├── config.py                        # Configuration settings
├── utils.py                         # Utility functions
├── requirements.txt                 # Dependencies
└── data/                            # Dataset
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
python run_experiments.py --student 12852 --methods MIRROR

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

## Results

See paper Section 4 for detailed experimental results and analysis.