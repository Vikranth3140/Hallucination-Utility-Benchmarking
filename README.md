# Hallucination Utility Benchmarking

A comprehensive framework for evaluating the utility of hallucinations in Large Language Model (LLM) outputs across different task types. This project introduces a novel perspective on hallucinations by classifying them based on their utility rather than just their presence.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate LLM Outputs](#1-generate-llm-outputs)
  - [2. Annotate Outputs](#2-annotate-outputs)
  - [3. Train Classifier](#3-train-classifier)
  <!-- - [4. Evaluate Results](#4-evaluate-results) -->
- [Methodology](#methodology)
- [Data Schema](#data-schema)
- [Configuration](#configuration)

## Overview

Not all hallucinations are created equal. While traditional approaches treat all hallucinations as errors, this framework recognizes that **hallucinations can sometimes be useful** depending on the context and task type. 

This project provides:
- A pipeline to generate LLM outputs under different prompting modes (SAFE vs PRESSURED)
- Automated hallucination utility annotation using judge models
- Machine learning models to classify hallucination utility
- Comprehensive evaluation metrics

## Key Concepts

### Hallucination Utility Labels

The framework categorizes hallucinations into three utility classes:

- **U+ (Useful)**: The response contains hallucination AND it enhances the task's goal
  - Example: Creative brainstorming where fabricated ideas spark innovation
  
- **U0 (Neutral)**: The response contains NO hallucination OR the hallucination neither helps nor harms
  - Example: Minor factual inaccuracies that don't affect the outcome
  
- **U- (Harmful)**: The response contains hallucination AND it misleads, confuses, or harms the task outcome
  - Example: Fabricated medical facts in a health advisory context

### Task Types

The framework evaluates hallucinations across three task categories:

1. **Factual**: Tasks requiring accurate, verifiable information
2. **Creative**: Tasks benefiting from imagination and novel ideas
3. **Brainstorm**: Tasks where exploratory thinking is valuable

### Generation Modes

- **SAFE Mode**: Model answers normally, admits uncertainty when unsure
- **PRESSURED Mode**: Model is instructed to give confident answers without refusing or expressing limitations

## Features

- **Multi-Model Support**: Test multiple LLMs simultaneously (Llama, Mistral, Qwen, etc.)
- **NVIDIA NIM Integration**: Leverages NVIDIA's NIM API for efficient inference
- **Automated Annotation**: Uses judge models to label hallucination utility automatically
- **ML Classification**: Train neural network classifiers to predict hallucination utility
- **Flexible Configuration**: Easy-to-configure prompts, models, and generation modes
- **Robust Error Handling**: Retry logic, exponential backoff, and graceful degradation
- **Comprehensive Logging**: Track generation, annotation, and training progress

## Project Structure

```
hallucination-benchmarking/
├── data/                           # Data directory
│   ├── prompts.json               # Input prompts for all task types
│   ├── raw_outputs_*.jsonl       # Generated model outputs
│   ├── labeled_*.jsonl           # Annotated outputs with utility labels
│   └── helper data/              # Additional dataset variants
├── src/                           # Source code
│   ├── generate.py               # Generate LLM outputs
│   ├── annotate.py               # Annotate with utility labels
│   ├── train.py                  # Train classification model
│   ├── llm_clients.py            # LLM API clients
│   ├── prompts.py                # Prompt templates
│   └── schema.py                 # Data schemas (Pydantic models)
├── requirements.txt              # Python dependencies
├── LICENSE                       # License file
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA API key (for NIM access)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vikranth3140/Hallucination-Utility-Benchmarking.git
   cd Hallucination-Utility-Benchmarking
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   NVIDIA_API_KEY=your_nvidia_api_key_here
   NIM_BASE_URL=https://integrate.api.nvidia.com/v1  # Optional, this is the default
   ```

## Usage

The pipeline consists of four main stages:

### 1. Generate LLM Outputs

Generate responses from multiple LLMs for your prompts:

```bash
python -m src.generate
```

**Configuration** (in [src/generate.py](src/generate.py)):
- Edit `MODE_BY_TASK` to set SAFE or PRESSURED mode per task type
- Add models to the `models` list
- Specify output path (`raw_outputs_SAFE.jsonl` or `raw_outputs_PRESSURED.jsonl`)

**Example model configuration:**
```python
models = [
    ("nim-llama", NIMClient(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.7,
        max_tokens=512,
    )),
    ("nim-mistral", NIMClient(
        model="mistralai/mistral-7b-instruct-v0.2",
        temperature=0.7,
        max_tokens=512,
    )),
]
```

### 2. Annotate Outputs

Use a judge model to automatically label hallucination utility:

```bash
python -m src.annotate
```

**Configuration** (in [src/annotate.py](src/annotate.py)):
- Set `in_path` to your raw outputs file
- Set `out_path` for labeled outputs
- Configure judge model (default: `meta/llama-3.1-70b-instruct`)

**The judge model evaluates:**
1. Does the response contain hallucination?
2. If yes, what is its utility (U+, U0, or U-)?

### 3. Train Classifier

Train a neural network to predict hallucination utility:

```bash
python -m src.train
```

**Features:**
- Uses sentence embeddings (`all-mpnet-base-v2`)
- Multi-layer perceptron (MLP) classifier
- Automatic train/validation/test split (70/15/15)
- Stratified sampling when possible
- Saves best model to `data/cahr_mlp.pt`

**Output:**
- Training progress with loss metrics
- Classification report (precision, recall, F1)
- Confusion matrix
- Trained model weights

<!-- ### 4. Evaluate Results

```bash
python -m src.eval
``` -->

## Methodology

### 1. Prompt Design

The framework uses two prompt templates:

**Generation Prompt** ([src/prompts.py](src/prompts.py)):
- Instructs the model on behavior (SAFE vs PRESSURED mode)
- Includes task type and user prompt
- Standardized output format

**Judge Prompt** ([src/prompts.py](src/prompts.py)):
- Clear hallucination and utility definitions
- Step-by-step evaluation instructions
- Structured JSON output

### 2. Data Collection

- Multiple models generate responses for each prompt
- Both SAFE and PRESSURED modes are tested
- All metadata (temperature, mode, model) is preserved

### 3. Annotation Pipeline

- Judge model evaluates each response
- Hallucination detection first, then utility assessment
- JSON parsing with fallback handling
- Label normalization (U+, U0, U-)

### 4. Model Training

- Sentence embeddings capture semantic meaning
- MLP learns utility classification
- Cross-entropy loss with AdamW optimizer
- Early stopping based on validation loss

## Data Schema

### PromptItem
```python
{
  "task_type": "factual" | "creative" | "brainstorm",
  "prompt_id": str,
  "prompt": str
}
```

### ModelOutput
```python
{
  "task_type": str,
  "prompt_id": str,
  "prompt": str,
  "model_name": str,
  "response": str,
  "meta": {
    "temperature": float,
    "mode": str
  }
}
```

### LabeledExample
```python
{
  "task_type": str,
  "prompt_id": str,
  "prompt": str,
  "model_name": str,
  "response": str,
  "utility_label": "U+" | "U0" | "U-",
  "rationale": str,
  "judge_model": str,
  "meta": dict
}
```

## Configuration

### Model Configuration

Edit [src/llm_clients.py](src/llm_clients.py) to:
- Change API base URL
- Adjust timeout settings
- Modify retry logic and backoff strategy

### Prompt Templates

Modify [src/prompts.py](src/prompts.py) to:
- Customize generation instructions
- Adjust judge evaluation criteria
- Change output formats

### Training Hyperparameters

Adjust in [src/train.py](src/train.py):
- Embedding model (`SentenceTransformer`)
- MLP architecture (hidden dimensions)
- Learning rate, epochs, batch size
- Train/val/test split ratios


