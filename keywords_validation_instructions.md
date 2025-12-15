# Keyword Validation Annotation Instructions

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Arc-Celt/potato.git
cd potato
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running Annotation Tasks

Three separate configuration files are available, one for each model.

### GPT-5 Mini

```bash
python potato/flask_server.py start configs/keywords_annotation_gpt5_mini.yaml -p 8000
```

**Access:** <http://localhost:8000>

### Qwen3 30B A3B Instruct FP8

```bash
python potato/flask_server.py start configs/keywords_annotation_qwen3_30b_a3b_instruct_fp8.yaml -p 8001
```

**Access:** <http://localhost:8001>

### Qwen3 32B FP8

```bash
python potato/flask_server.py start configs/keywords_annotation_qwen3_32b_fp8.yaml -p 8002
```

**Access:** <http://localhost:8002>

## Annotation Output

User annotations will be saved to the following output directories:

- **GPT-5 Mini:** `annotation_output/keyword_annotation_gpt5_mini/`
- **Qwen3 30B A3B Instruct FP8:** `annotation_output/keyword_annotation_qwen3_30b_a3b_instruct_fp8/`
- **Qwen3 32B FP8:** `annotation_output/keyword_annotation_qwen3_32b_fp8/`

Once the annotation is complete, the data can be used for evaluation.
