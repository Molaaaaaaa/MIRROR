# MIRROR Framework

**Memory-Integrated Reasoning and Retrieval for Longitudinal Response Prediction**

## Architecture Overview (Paper Figure 2)

```
+-----------------------------------------------------------------------------+
|                     Offline Stage (Cache Construction)                        |
+---------------------+-------------------------+-----------------------------+
| 1. RER              | 2. LTE                  | 3. KG                       |
|                     |                         |                             |
| H_u -> D_u          | P^static (Demographics) | Nodes: Domain -> Q -> Opt   |
| (Q-A-Year docs)     | P^domain (Domain trend) | Edges:                      |
| ChromaDB storage    | P^year (Item trace)     |   - contains (schema)       |
|                     | P^change (Sudden shift) |   - correlates (Pearson)    |
|                     |                         |   - similar_to (embedding)  |
+---------------------+-------------------------+-----------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|                     Online Stage (Inference)                                  |
|  4. Consistency-aware Prediction                                              |
|                                                                               |
|  CTX_{u,q} = (q, Y_q, E_{u,q}, P_{u,q}, C_q)                                |
|                                                                               |
|  - q: Target question                                                         |
|  - Y_q: Valid option range (from KG schema)                                   |
|  - E_{u,q}: Retrieved evidence (from RER)                                     |
|  - P_{u,q}: Persona profiles (from LTE)                                       |
|  - C_q: KG consistency constraints                                            |
|                                                                               |
|  Single-step constrained inference -> prediction + explanation                |
+-----------------------------------------------------------------------------+
```

## Dataset

The dataset is derived from the **Korean Children and Youth Panel Survey (KCYPS)**, a nationally representative longitudinal study tracking adolescents from 2018 to 2023.

### Data Structure

```
data/
├── schema.json           # Survey question option schema (category -> options)
└── students/
    ├── {student_id}/
    │   ├── {student_id}_2018.csv
    │   ├── {student_id}_2019.csv
    │   ├── {student_id}_2020.csv
    │   ├── {student_id}_2021.csv
    │   ├── {student_id}_2022.csv
    │   └── {student_id}_2023.csv  # Ground truth (target year)
    └── ... (100 students)
```

### CSV Format

Each yearly CSV contains survey responses with the following columns:

| Column | Description |
|--------|-------------|
| `응답자` | Respondent grade level |
| `응답년도` | Survey year (2018-2023) |
| `코드북 일련번호` | Codebook serial number |
| `설문 문항` | Survey question text |
| `응답 내용` | Response value |

### Survey Categories

The dataset covers diverse behavioral and psychological domains including:
- **Target domains**: Aggression, School Violence, Delinquency
- **Related domains**: Social withdrawal, Depression, Attention, Teacher/Peer relationships, Academic motivation, Self-esteem, and more

### Experimental Settings

| Setting | Description |
|---------|-------------|
| **S1** (Full-history) | All 2018-2022 responses available for prediction |
| **S2** (Violence-blinded) | Excludes Aggression, School Violence, Delinquency from history |
| **S3** (S2 + Aggression) | S2 with Aggression restored |


## Requirements

- Python 3.10+
- NVIDIA GPU (recommended for local LLM)

```bash
pip install langchain-ollama langchain-openai langchain-huggingface langchain-chroma
pip install sentence-transformers chromadb pandas tqdm
```

## LLM Backend Configuration

The framework supports three LLM providers. Set in `config.py` or via environment variables:

### 1. Ollama (Local, default)
```bash
ollama pull qwen3:14b-q8_0
# config.py: LLM_PROVIDER = "ollama", OLLAMA_MODEL = "qwen3:14b-q8_0"
```

### 2. OpenAI API
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=gpt-4o-mini-2024-07-18
```

### 3. DeepSeek API
```bash
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your_api_key
export DEEPSEEK_MODEL=deepseek-chat           # DeepSeek-V3
```

## Quick Start

```bash
# 1. Build LTE (Longitudinal Trend Extraction) for all students
python build_ltm.py --all --workers 2

# 2. Build KG (Knowledge Graph) for all students
python build_kg.py --all --workers 2

# 3. Run MIRROR prediction (Setting S1)
python run_experiments.py --all --methods MIRROR

# 4. Evaluate results (micro + domain-wise macro accuracy)
python calc_acc.py results/MIRROR_S1
```

## Usage

### Run Experiments

```bash
# Full ablation study (Table 5)
python run_experiments.py --all --ablation

# Individual methods
python run_experiments.py --all --methods LLM_only
python run_experiments.py --all --methods RER
python run_experiments.py --all --methods RER_LTE
python run_experiments.py --all --methods RER_KG
python run_experiments.py --all --methods MIRROR

# Different settings (S1/S2/S3)
python run_experiments.py --all --methods MIRROR                    # S1: Full history
python run_experiments.py --all --methods MIRROR --exclude-target   # S2: Violence-blinded
python run_experiments.py --all --methods MIRROR --exclude-partial  # S3: S2 + Aggression

# Single student (for debugging)
python run_experiments.py --student 12852 --methods MIRROR --debug
```

### Evaluate Results

```bash
# Micro accuracy + Domain-wise macro accuracy (Table 6)
python calc_acc.py results/MIRROR_S1
```

## Components

### 1. RER (Retrospective Evidence Retrieval)

**Paper Equation (1):** `E_{u,q} = R(z_q, D_u, K)`

- Converts history `H_u` into Q-A-Year documents
- Dense retrieval using `PwC-Embedding_expr` embedding model
- Returns top-K relevant historical evidence

### 2. LTE (Longitudinal Trend Extraction)

**Persona Components:**
- `P^static_u`: Demographics (gender, birth year, region)
- `P^domain_u`: Domain-level trend summary
- `P^year_u`: 5-year temporal trace per question
- `P^change_u`: Sudden shift detection (Z-score > 2.0)

### 3. KG (Knowledge Graph)

**Graph Structure (G = (V, E)):**
- **Nodes (V):** Domain, Question, Option nodes
- **Edges (E):**
  - Inclusion: domain -> question -> option (schema)
  - Association: Pearson correlation (|r| >= 0.3), semantic similarity (>= 0.5)

### 4. Consistency-aware Prediction

- Constructs CTX_{u,q} dynamically per question
- Places sudden shift warning at top of prompt
- Includes KG valid option range constraint
- Generates prediction with brief explanation

## Method Configurations (Table 5)

| Method | RER | LTE | KG | Description |
|--------|-----|-----|-----|-------------|
| `LLM_only` | - | - | - | Plain LLM baseline |
| `RER` | O | - | - | + Evidence retrieval |
| `RER_LTE` | O | O | - | + Longitudinal trends |
| `RER_KG` | O | - | O | + Knowledge graph |
| `MIRROR` | O | O | O | Full framework |

## File Structure

```
MIRROR/
├── config.py                 # Configuration (LLM provider, paths, hyperparameters)
├── mirror_framework.py       # Core prediction framework (online stage)
├── build_ltm.py              # LTE builder (offline stage)
├── build_kg.py               # KG builder (offline stage)
├── tools.py                  # RER toolkit (vectorstore, embedding, retrieval)
├── run_experiments.py         # Experiment runner (S1/S2/S3 settings, ablation)
├── calc_acc.py               # Accuracy calculator (micro, domain-wise macro)
├── utils.py                  # Data loading, output cleaning, evaluation utilities
├── data_constants.py         # Dataset-specific constants (categories, keywords)
├── data/
│   ├── schema.json           # Survey question option schema
│   └── students/             # Student longitudinal data (100 students x 6 years)
├── mirror_memory/            # Pre-built LTE and KG data (offline stage output)
├── chroma_cache/             # Vectorstore cache for RER
└── results/                  # Experiment results
```
