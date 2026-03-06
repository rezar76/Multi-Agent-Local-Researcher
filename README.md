# Multi-Agent Research Pipeline

> A fully local, domain-agnostic AI research assistant powered by **Ollama** and **6 specialised agents** that plans, searches the web, analyses sources in chunks, writes a detailed report, and derives a formal mathematical framework — all from a single command.

![Pipeline Architecture](pipeline_architecture.png)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
  - [Agent Breakdown](#agent-breakdown)
  - [Data Flow](#data-flow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Interactive Mode](#interactive-mode)
  - [Change the Model](#change-the-model)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [How It Works — Deep Dive](#how-it-works--deep-dive)
- [Visualisation](#visualisation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a **Multi-Agent Research Pipeline** that takes any natural-language research query and autonomously:

1. Builds a structured research plan  
2. Extracts all relevant technical entities  
3. Searches the web for real-world sources  
4. Analyses them across multiple LLM passes  
5. Generates a comprehensive multi-section research report (`.md`)  
6. Derives a rigorous mathematical framework in LaTeX (`.tex`)  

Everything runs **100% locally** — no API keys, no cloud services. Just Ollama and an internet connection for DuckDuckGo searches.

---

## Features

- **Domain-agnostic** — works on any topic: signal processing, biology, finance, machine learning, law, etc.
- **No context contamination** — each agent call is fully stateless; no stale history leaks between runs
- **Multi-pass analysis** — web results are split into overlapping 5 000-word chunks and processed iteratively before synthesis
- **Section-by-section report generation** — each report section is a dedicated LLM call, ensuring depth and detail (≥ 300 words per section)
- **Adaptive LaTeX formalisation** — first detects the correct mathematical framework for the domain, then writes each section independently
- **Unique output filenames** — based on a URL-safe slug + MD5 hash of the query, so repeated runs never overwrite each other
- **Rich terminal UI** — colour-coded progress, rules, and step indicators via the `rich` library

---

## Architecture

```
[ User Query ]
      │
      ▼
┌─────────────────┐     ┌─────────────────────┐
│  PlanningAgent  │────▶│  QueryAnalysisAgent  │
│  (Architect)    │     │  (Librarian)         │
└────────┬────────┘     └──────────┬──────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐     ┌─────────────────────┐
│ DataGatherAgent │────▶│   AnalysisAgent      │
│ (Web Researcher)│     │ (Multi-Pass Synth.)  │
└────────┬────────┘     └──────────┬──────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐     ┌─────────────────────┐
│   ReportAgent   │     │  MathTheoristAgent   │
│ (Tech Writer)   │     │  (Formaliser)        │
└────────┬────────┘     └──────────┬──────────┘
         │                         │
         ▼                         ▼
  research_report_       theoretical_solution_
     <slug>.md               <slug>.tex
```

### Agent Breakdown

| # | Agent | Role | LLM Calls |
|---|-------|------|-----------|
| 1 | **PlanningAgent** | Detects the domain, identifies key challenges, generates a 4–5 step execution roadmap | 1 |
| 2 | **QueryAnalysisAgent** | Extracts primary topic, sub-topics, key concepts, and technologies from the raw query | 1 |
| 3 | **DataGatherAgent** | Generates 5 optimised search queries, fetches up to 50 DuckDuckGo snippets | 1 |
| 4 | **AnalysisAgent** | Per-chunk extraction → cross-chunk synthesis → deep-dive review (3 passes minimum) | N+2 |
| 5 | **ReportAgent** | Writes 6 report sections independently: Summary, Parameters, Methodology, Analysis, Recommendations, Conclusion | 6 |
| 6 | **MathTheoristAgent** | Identifies math framework → writes Problem Formulation, System Model, Solution, Complexity Analysis | 5 |

> **Total LLM calls per pipeline run: ~17+ (scales with number of context chunks)**

### Data Flow

```
query ──► ExecutionPlan + QueryEntities ──► research_data{raw_context}
                                                    │
                                          ┌─────────▼──────────┐
                                          │  chunk_text()       │
                                          │  5 000-word chunks  │
                                          │  80-word overlap    │
                                          └─────────┬──────────┘
                                                    │
                                         per-chunk analysis × N
                                                    │
                                           synthesis pass (1×)
                                                    │
                                           deep-dive pass (1×)
                                                    │
                                    ┌───────────────┴────────────────┐
                                    │                                │
                             ReportAgent                    MathTheoristAgent
                          (6 section calls)                 (5 section calls)
                                    │                                │
                             *.md report                       *.tex paper
```

---

## Requirements

- **Python** 3.10+
- **Ollama** with at least one chat model pulled (e.g. `qwen3.5:9b`)
- Internet connection (for DuckDuckGo searches)

### Python dependencies

```
openai
pydantic
ddgs
rich
matplotlib        # only for visualize_pipeline.py
```

---

## Installation

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a model

```bash
ollama pull qwen3.5:9b
```

> Any Ollama-compatible chat model works. Larger models (14b, 32b) produce higher quality output.

### 3. Clone this repository

```bash
git clone https://github.com/your-username/multiagent-research.git
cd multiagent-research
```

### 4. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 5. Install Python dependencies

```bash
pip install openai pydantic ddgs rich matplotlib
```

---

## Usage

Make sure Ollama is running before executing the pipeline:

```bash
ollama serve   # if not already running as a service
```

### Basic Usage

Pass your research query as a command-line argument:

```bash
python multiagent_research.py "Investigate transfer learning approaches for low-resource NLP tasks"
```

### Interactive Mode

Run without arguments and enter the query when prompted:

```bash
python multiagent_research.py
# Enter your research query: <type here>
```

### Change the Model

Use the `--model` flag to switch to any Ollama model:

```bash
python multiagent_research.py "Your query" --model llama3.2:3b
python multiagent_research.py "Your query" --model mistral:7b
python multiagent_research.py "Your query" --model qwen2.5:14b
```

### Example full run

```
════════════════════════════════════════
Multi-Agent Research Pipeline
════════════════════════════════════════
Query: Investigate the use of dictionary learning for sparse signal recovery
Model: qwen3.5:9b
Outputs: research_report_investigate_the_use_of_dic_11652a.md  |  theoretical_solution_investigate_the_use_of_dic_11652a.tex

── Step 1 / 5 — Planning ──────────────────
Domain detected: Signal Processing / Machine Learning
Strategy: A comparative study of dictionary learning algorithms...
Key challenges: ['coherence of learned atoms', 'scalability to high dimensions', ...]

── Step 2 / 5 — Entity Extraction ─────────
Primary topic:  Dictionary Learning for Sparse Signal Recovery
Domain:         Signal Processing
Sub-topics:     ['K-SVD', 'OMP', 'LASSO', 'compressed sensing']
Key concepts:   ['sparsity', 'mutual coherence', 'RIP condition']
Tools/Tech:     ['numpy', 'sklearn', 'PyTorch']

── Step 3 / 5 — Web Research ──────────────
  Collecting 5 queries × 10 results...
Collected 47 snippets from 5 queries.

── Step 4 / 5 — Multi-Pass Analysis ───────
  Processing 9 context chunks...
  → Chunk 1/9 ... → Chunk 9/9
  → Synthesising across chunks...
  → Deep-dive pass...
Analysis complete.

── Step 5a / 5 — Report Generation ────────
  → Executive Summary...
  → Research Parameters...
  → Methodology...
  → Technical Analysis...
  → Recommendations...
  → Conclusion...
Report saved: research_report_investigate_the_use_of_dic_11652a.md

── Step 5b / 5 — Mathematical Formalisation
  → Identifying mathematical framework...
     Framework: Convex Optimisation and Sparse Representation
  → Problem Formulation...
  → System Model...
  → Proposed Solution...
  → Complexity Analysis...
LaTeX theory saved: theoretical_solution_investigate_the_use_of_dic_11652a.tex

══════════════════ PIPELINE COMPLETE ══════
  Report  → research_report_investigate_the_use_of_dic_11652a.md
  Theory  → theoretical_solution_investigate_the_use_of_dic_11652a.tex
```

---

## Output Files

Every run produces two files, both named with a **unique slug** derived from your query so runs never overwrite each other:

### `research_report_<slug>.md`

A structured Markdown research report with the following sections:

| Section | Content |
|---------|---------|
| **Executive Summary** | 3–5 sentence overview of key findings and top recommendation |
| **Research Parameters** | Primary topic, sub-topics, key concepts, tools/technologies |
| **Methodology** | Description of the multi-agent pipeline and research strategy |
| **Technical Analysis** | Full comparative evaluation of methods with pros/cons, benchmarks, rankings |
| **Recommendations** | 5–7 actionable items with effort estimates and dependencies |
| **Conclusion** | Summary of findings, open questions, and significance |

### `theoretical_solution_<slug>.tex`

A standalone, compilable LaTeX document containing:

| Section | Content |
|---------|---------|
| **Abstract** | Auto-generated from domain and sub-topics |
| **Problem Formulation** | Formal objective, constraints, and notation |
| **System Model** | Governing equations derived from first principles |
| **Proposed Theoretical Solution** | Step-by-step derivation with proofs |
| **Computational Complexity Analysis** | Big-O analysis with hardware context |

To compile the LaTeX to PDF:

```bash
pdflatex theoretical_solution_<slug>.tex
# or, for full bibliography and cross-reference support:
latexmk -pdf theoretical_solution_<slug>.tex
```

---

## Project Structure

```
multiagent-research/
│
├── multiagent_research.py      # Main pipeline — run this
├── visualize_pipeline.py       # Generates the architecture diagram
├── pipeline_architecture.png   # Architecture diagram (300 dpi)
├── MultiAgent.ipynb            # Original Jupyter notebook (development)
│
├── research_report_*.md        # Generated reports (gitignored recommended)
└── theoretical_solution_*.tex  # Generated LaTeX papers (gitignored recommended)
```


---

## How It Works — Deep Dive

### Stateless LLM calls

All agents share a single `llm()` helper function. Every call creates a **fresh** messages array with only `system` + `user` roles — no chat history is ever accumulated. This eliminates cross-run domain contamination (the main cause of a radar-specialised model generating radar output for unrelated topics).

### Context chunking

Web search results can be very long. The `chunk_text()` function splits the raw context into 5 000-word chunks with an 80-word overlap at each boundary. Each chunk is analysed independently, then a synthesis pass merges the results.

### Slug-based unique filenames

```python
slug = re.sub(r"[^\w\s-]", "", query.lower())[:40]
slug += "_" + hashlib.md5(query.encode()).hexdigest()[:6]
```

Two similar queries (e.g. "radar clutter" vs "radar clutter mitigation") produce different slugs because the hash is computed from the full query string.

### Domain detection

The `PlanningAgent` and `QueryAnalysisAgent` both independently detect the **domain** (e.g. "Signal Processing", "Computational Biology", "Quantitative Finance") from the query text. This domain label is injected into every subsequent agent's system prompt, keeping all analysis on-topic.

---

## Visualisation

Regenerate the architecture diagram at any time:

```bash
python visualize_pipeline.py
# Outputs: pipeline_architecture.png (300 dpi, ~6600×9000 px)
```

---

## Troubleshooting

### `Connection refused` on port 11434

Ollama is not running. Start it with:

```bash
ollama serve
```

### `model not found`

The model hasn't been pulled yet:

```bash
ollama pull qwen3.5:9b
```

### Zero web snippets collected

- Check your internet connection
- DuckDuckGo occasionally rate-limits. Wait 30–60 seconds and retry
- Increase the sleep delay in `DataGatherAgent.gather_data()` (default `1.5s`)

### LaTeX compilation fails

Ensure the required LaTeX packages are installed:

```bash
# Ubuntu/Debian
sudo apt install texlive-full

# macOS (Homebrew)
brew install --cask mactex
```

### Output is short or off-topic

- Use a **larger model**: `--model qwen2.5:14b` or `--model qwen3.5:32b`
- Larger models follow the structured prompts more reliably and produce richer text
- Models < 7B may struggle with JSON schema compliance

---

## Contributing

Contributions are welcome! Ideas for improvement:

- [ ] Add async/parallel agent execution
- [ ] Support for academic paper search (arXiv, Semantic Scholar API)
- [ ] Citation tracking with proper BibTeX generation

Please open an issue before submitting large PRs.

---


---

<div align="center">
  <sub>Built with Ollama · DuckDuckGo DDGS · Pydantic · Rich · Matplotlib</sub>
</div>
