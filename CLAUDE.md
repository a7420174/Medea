# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medea is an AI-powered multi-agent system for therapeutic discovery through multi-omics analysis. It combines three specialized agents (Research Planning, Experiment Analysis, Literature Reasoning) that run in parallel, with results synthesized via a multi-round panel discussion among multiple LLMs.

## Setup & Installation

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Configuration: copy `env_template.txt` to `.env` and set required variables (`MEDEADB_PATH`, `BACKBONE_LLM`, `SEED`, and at least one LLM provider API key).

MedeaDB (large biology database): `git clone https://huggingface.co/datasets/mims-harvard/MedeaDB`

## Running

```bash
# Library usage
from medea import medea, AgentLLM, LLMConfig, ResearchPlanning, Analysis, LiteratureReasoning

# CLI evaluation
python main.py --task targetID --disease ra --temperature 0.5 --debate-rounds 3
```

## Architecture

### Core Flow (`medea/core.py`)

Three entry points: `medea()` (full system), `experiment_analysis()` (plan + analysis only), `literature_reasoning()` (literature only).

`medea()` runs experiment_analysis and literature_reasoning in **parallel via multiprocessing**, then synthesizes via `multi_round_discussion()`.

### Three Agent Modules (`medea/modules/`)

Each agent extends `BaseAgent` from `langchain_agents.py` and contains a sequence of `BaseAction` steps:

1. **ResearchPlanning** (`research_planning.py`): ResearchPlanDraft → ContextVerification → IntegrityVerification (iterative, max_iter=2). Outputs a `Proposal` object.
2. **Analysis** (`experiment_analysis.py`): CodeGenerator → AnalysisExecution (subprocess) → CodeDebug (iterative) → AnalysisQualityChecker (max_iter=2). Generates and runs Python code for single-cell analysis.
3. **LiteratureReasoning** (`literature_reasoning.py`): LiteratureSearch → PaperJudge (FlagEmbedding reranker) → OpenScholarReasoning. Queries Semantic Scholar, OpenAlex, PubMed.

### Panel Discussion (`medea/modules/discussion.py`)

`multi_round_discussion()` orchestrates multi-round debate among configurable panelist LLMs (default: gemini-2.5-flash, o3-mini, backbone LLM).

### LLM Infrastructure

- **AgentLLM** (`agent_llms.py`): Unified LLM interface wrapping `chat_completion()`.
- **chat_completion** (`tool_space/gpt_utils.py`): Multi-provider backend. Auto-detects provider from env vars: OpenRouter → Azure → Gemini → Anthropic → fallback error.
- **env_utils.py**: Environment variable management with provider auto-detection.

### Tool Space (`medea/tool_space/`)

External database integrations: DepMap, EnrichR, HumanBase, Human Protein Atlas, OpenAlex, Semantic Scholar, PubMed, TranscriptFormer, COMPASS. Tool metadata in `tool_info.json`. LLM selects relevant tools per query via `action_functions.py`.

### Key Data Classes (`medea/modules/utils.py`)

- `Proposal`: Research plan with disease, genes, cell types, analysis steps
- `CodeSnippet`: Generated code with execution results

### Prompts

All prompt templates in `prompt_template.py` (~1100 lines). Prompt assembly utilities in `BasePrompt.py` and `prompt_utils.py`.

## Public API (`medea/__init__.py`)

Core functions, all three agent classes, all action classes, data classes, and `multi_round_discussion` are exported from the top-level `medea` package.
