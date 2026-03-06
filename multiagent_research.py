#!/usr/bin/env python3
"""
Multi-Agent Research Pipeline
Usage:
    python multiagent_research.py "Your research query here"
    python multiagent_research.py  # Will prompt for input interactively
    python multiagent_research.py "Your query" --model qwen3.5:9b
"""

import json
import time
import re
import sys
import argparse
import hashlib
import textwrap
from datetime import datetime
from typing import List, Optional, Union, Dict

from openai import OpenAI
from pydantic import BaseModel, Field, validator
from ddgs import DDGS
from rich import print as rich_print
from rich.progress import Progress
from rich.rule import Rule

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def make_slug(query: str, max_len: int = 40) -> str:
    """Generate a filesystem-safe slug from a query string."""
    slug = re.sub(r"[^\w\s-]", "", query.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    slug = slug[:max_len]
    short_hash = hashlib.md5(query.encode()).hexdigest()[:6]
    return f"{slug}_{short_hash}"


def llm(client: OpenAI, model: str, system: str, user: str,
        temperature: float = 0.2, json_mode: bool = False) -> str:
    """Centralised LLM call. Every agent uses this — no stale context between runs."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": textwrap.dedent(system).strip()},
            {"role": "user",   "content": textwrap.dedent(user).strip()},
        ],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split context into overlapping chunks so large results don't get truncated."""
    words = text.split()
    chunks, current, size = [], [], 0
    for w in words:
        current.append(w)
        size += len(w) + 1
        if size >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-80:]   # 80-word overlap for continuity
            size = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

class PlanStep(BaseModel):
    step_id: int
    agent_name: str
    description: str
    input_requirements: List[str] = Field(default_factory=list)
    expected_output: str
    status: str = "PENDING"


class ExecutionPlan(BaseModel):
    original_query: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_researchable: bool
    research_strategy: str
    domain: str = "General Research"
    key_challenges: List[str] = Field(default_factory=list)
    steps: List[PlanStep]
    requires_web_search: bool = True


class QueryEntities(BaseModel):
    primary_topic: str
    domain: str = "General"
    sub_topics: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    technologies_or_tools: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    time_period: Optional[Union[str, List[str]]] = None
    output_format: str = "Technical Analysis"
    research_theme: str = "General Research"

    @validator("time_period", "output_format", pre=True)
    def coerce_to_string(cls, v):
        if isinstance(v, list):
            return ", ".join(map(str, v)) if v else None
        return v


# ─────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────

class PlanningAgent:
    """Creates a structured, domain-aware research roadmap for any topic."""

    def __init__(self, client: OpenAI, model_name: str = "qwen3.5:9b"):
        self.client = client
        self.model = model_name

    def create_plan(self, user_query: str) -> ExecutionPlan:
        system = f"""
            You are a Senior Research Architect. Analyse ANY research query and produce
            a precise, domain-aware execution roadmap.

            OUTPUT — a single JSON object matching this EXACT schema:
            {{
                "original_query":      "{user_query}",
                "is_researchable":     true,
                "research_strategy":   "<2-3 sentence approach tailored to THIS specific topic>",
                "domain":              "<detected domain, e.g. Signal Processing, Finance, Biology>",
                "key_challenges":      ["<challenge 1>", "<challenge 2>", "<challenge 3>"],
                "requires_web_search": true,
                "steps": [
                    {{
                        "step_id": 1,
                        "agent_name": "DataGatherAgent",
                        "description": "<specific task>",
                        "input_requirements": ["<req>"],
                        "expected_output": "<artifact>"
                    }}
                ]
            }}

            RULES:
            - steps: 4-5 items covering Gather → Analyse → DeepDive → Synthesise → MathFormalise
            - research_strategy and key_challenges must be specific to the EXACT query topic
            - do NOT borrow language from radar, finance, or any other domain
              unless those words explicitly appear in the query
        """
        raw = llm(self.client, self.model, system,
                  f"Architect a research plan for:\n\n{user_query}",
                  temperature=0.1, json_mode=True)
        data = json.loads(raw)
        data.setdefault("original_query", user_query)
        for step in data.get("steps", []):
            if "agent_type" in step: step["agent_name"] = step.pop("agent_type")
            if "task"       in step: step["description"] = step.pop("task")
        return ExecutionPlan(**data)


class QueryAnalysisAgent:
    """Extracts rich, domain-specific entities from any query without domain bias."""

    def __init__(self, client: OpenAI, model_name: str = "qwen3.5:9b"):
        self.client = client
        self.model = model_name

    def analyze(self, user_query: str) -> QueryEntities:
        system = """
            You are an expert Research Librarian. Extract ALL technical entities
            from the user query. Be exhaustive and strictly domain-accurate.

            Return ONLY a JSON object:
            {
                "primary_topic":         "<core subject>",
                "domain":                "<academic/industry domain>",
                "sub_topics":            ["<specific technical problem 1>", ...],
                "key_concepts":          ["<important concept or method>", ...],
                "technologies_or_tools": ["<hardware / software / framework>", ...],
                "locations":             ["<geographic or institutional context>"],
                "time_period":           "<temporal scope or null>",
                "output_format":         "<desired deliverable>",
                "research_theme":        "<overarching research theme>"
            }

            CRITICAL: Extract entities ONLY from the query text itself.
            Do NOT add domain-specific jargon that the user did not write.
        """
        raw = llm(self.client, self.model, system, user_query,
                  temperature=0.05, json_mode=True)
        return QueryEntities(**json.loads(raw))


class DataGatherAgent:
    """Generates 5 targeted search queries and gathers up to 50 web snippets."""

    def __init__(self, client: OpenAI, model_name: str = "qwen3.5:9b"):
        self.client = client
        self.model = model_name

    def _generate_queries(self, entities: QueryEntities) -> List[str]:
        system = f"""
            You are a Search Optimisation Expert for the domain: {entities.domain}.
            Generate 5 high-precision search strings to retrieve technical
            documentation, papers, and implementation guides.

            Target topic:    {entities.primary_topic}
            Sub-topics:      {entities.sub_topics}
            Key concepts:    {entities.key_concepts}
            Tools/Tech:      {entities.technologies_or_tools}

            Return JSON:
            {{"queries": ["<q1>", "<q2>", "<q3>", "<q4>", "<q5>"]}}

            QUERY STRATEGY:
            1. Official documentation or standards for the primary topic
            2. Core technical problem stated in the query
            3. Specific methods/algorithms mentioned in the query
            4. Comparative studies or benchmarks
            5. Recent advances or state-of-the-art (last 3 years)
        """
        raw = llm(self.client, self.model, system,
                  f"Generate search queries for: {entities.primary_topic}",
                  json_mode=True)
        return json.loads(raw).get("queries", [])

    def gather_data(self, entities: QueryEntities) -> dict:
        queries = self._generate_queries(entities)
        all_results = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Conducting Web Research...", total=len(queries))
            with DDGS() as ddgs:
                for query in queries:
                    try:
                        results = list(ddgs.text(query, max_results=10, backend="lite"))
                        all_results.extend(results)
                        progress.advance(task)
                        time.sleep(1.5)
                    except Exception as e:
                        rich_print(f"[yellow]Warning: '{query}' failed — {e}[/yellow]")

        context = "\n\n".join(
            f"SOURCE: {r.get('href','N/A')}\nTITLE: {r.get('title','N/A')}\nCONTENT: {r.get('body','N/A')}"
            for r in all_results
        )
        return {"search_queries": queries, "raw_context": context,
                "result_count": len(all_results)}


class AnalysisAgent:
    """Three-pass analysis: per-chunk extraction → synthesis → deep-dive.
    Processes context in chunks to avoid context window overflow and cross-run bleed."""

    def __init__(self, client: OpenAI, model_name: str = "qwen3.5:9b"):
        self.client = client
        self.model = model_name

    def _analyse_chunk(self, entities: QueryEntities, chunk: str,
                       idx: int, total: int) -> str:
        system = f"""
            You are a Senior Research Analyst specialising in: {entities.domain}.
            Topic: {entities.primary_topic}  |  Sub-topics: {entities.sub_topics}

            You are processing chunk {idx+1} of {total} from web search results.

            TASK:
            - Identify every distinct technical method, approach, or solution mentioned.
            - For each: summarise its mechanism, key findings, advantages, and limitations.
            - Note any numerical data, version numbers, benchmarks, or hardware specs cited.
            - Flag contradictions or open research questions.

            Be DETAILED. Use bullet points and sub-sections. Do not truncate.
            Only reference content present in this chunk.
        """
        return llm(self.client, self.model, system,
                   f"CONTEXT CHUNK:\n{chunk}", temperature=0.1)

    def _synthesise(self, entities: QueryEntities, partials: List[str]) -> str:
        combined = "\n\n---CHUNK BOUNDARY---\n\n".join(partials)
        system = f"""
            You are a Principal Research Scientist in: {entities.domain}.
            You have {len(partials)} partial analyses about: {entities.primary_topic}.

            SYNTHESIS TASK:
            1. Merge overlapping findings — eliminate duplicates.
            2. Identify the top 4-6 distinct methods/solutions with evidence.
            3. For each method produce:
               a. Full technical description (mechanism, algorithm, parameters)
               b. Quantitative performance data if available
               c. Pros and Cons table
               d. Feasibility for: {entities.technologies_or_tools}
            4. Rank methods by effectiveness, maturity, and practical applicability.
            5. Identify critical research gaps not addressed by existing work.
            6. Write a "Comparative Evaluation" section with ranking rationale.

            Minimum 3-4 paragraphs per method. Be comprehensive.
        """
        return llm(self.client, self.model, system,
                   f"PARTIAL ANALYSES:\n\n{combined}", temperature=0.1)

    def _deep_dive(self, entities: QueryEntities, synthesis: str) -> str:
        system = f"""
            You are a Deep Technical Reviewer in: {entities.domain}.
            You have a synthesis of research on: {entities.primary_topic}.

            DEEP DIVE TASK:
            1. Pick the top 2 most promising methods from the synthesis.
            2. For each, provide a step-by-step technical breakdown:
               - Input/output specification
               - Core algorithmic or procedural steps
               - Known failure modes and edge cases
               - Integration requirements with: {entities.technologies_or_tools}
            3. Propose one novel hybrid approach combining insights from ≥2 methods.
            4. Assess implementation complexity (Low/Medium/High) with justification.
            5. Identify 3 concrete follow-up experiments or validation steps.

            Minimum 600 words. Be rigorous and specific.
        """
        return llm(self.client, self.model, system,
                   f"SYNTHESIS:\n\n{synthesis}", temperature=0.15)

    def analyze_research(self, entities: QueryEntities, research_data: dict) -> str:
        context = research_data["raw_context"]
        if not context.strip():
            return "No web data was retrieved. Analysis cannot proceed."

        chunks = chunk_text(context, chunk_size=5000)
        rich_print(f"  [dim]Processing {len(chunks)} context chunk(s)...[/dim]")

        partials = []
        for i, chunk in enumerate(chunks):
            rich_print(f"  [dim]  → Chunk {i+1}/{len(chunks)}[/dim]")
            partials.append(self._analyse_chunk(entities, chunk, i, len(chunks)))

        rich_print("  [dim]  → Synthesising across chunks...[/dim]")
        synthesis = self._synthesise(entities, partials)

        rich_print("  [dim]  → Deep-dive pass...[/dim]")
        deep = self._deep_dive(entities, synthesis)

        return f"## SYNTHESIS\n\n{synthesis}\n\n---\n\n## DEEP-DIVE REVIEW\n\n{deep}"


class ReportAgent:
    """Generates each report section via a separate LLM call, then merges.
    This avoids truncation and produces far more detailed output."""

    def __init__(self, client: OpenAI, model_name: str = "qwen3.5:9b"):
        self.client = client
        self.model = model_name

    def _section(self, title: str, instructions: str,
                 context: str, entities: QueryEntities) -> str:
        system = f"""
            You are a Professional Technical Writer.
            Domain: {entities.domain}  |  Topic: {entities.primary_topic}
            Research theme: {entities.research_theme}

            Write ONLY the "{title}" section of a research report.
            {instructions}

            REQUIREMENTS:
            - Use formal academic/engineering English.
            - Be specific and detailed — no vague generalisations.
            - Minimum 300 words for this section.
            - Use sub-headings, bullet points, and numbered lists where appropriate.
            - Do NOT include any sections other than "{title}".
        """
        return llm(self.client, self.model, system,
                   f"SOURCE MATERIAL:\n\n{context}", temperature=0.25)

    def generate_final_report(self, plan: ExecutionPlan,
                               entities: QueryEntities, analysis: str) -> str:
        rich_print("  [dim]  → Executive Summary...[/dim]")
        summary = self._section(
            "EXECUTIVE SUMMARY",
            "Write 3-5 sentences: what was investigated, key findings, "
            "the top recommended approach, and the primary open challenge.",
            analysis, entities)

        rich_print("  [dim]  → Research Parameters...[/dim]")
        params = self._section(
            "RESEARCH PARAMETERS",
            f"Document: primary topic, sub-topics, key concepts, "
            f"tools/technologies ({entities.technologies_or_tools}), scope, and constraints. "
            "Explain why each parameter is relevant to the research.",
            f"Plan: {plan.research_strategy}\nEntities: {entities.model_dump()}",
            entities)

        rich_print("  [dim]  → Methodology...[/dim]")
        methodology = self._section(
            "METHODOLOGY",
            f"Describe the multi-agent pipeline: {[s.description for s in plan.steps]}. "
            "Explain how web data was gathered, chunked, and analysed. "
            "Justify the research strategy.",
            f"Plan: {plan.model_dump()}", entities)

        rich_print("  [dim]  → Technical Analysis...[/dim]")
        tech_section = self._section(
            "TECHNICAL ANALYSIS",
            "Present all methods identified, their comparative evaluation, "
            "deep-dive findings, and ranking. Include quantitative data where available. "
            "Describe trade-offs in detail.",
            analysis, entities)

        rich_print("  [dim]  → Recommendations...[/dim]")
        recs = self._section(
            "RECOMMENDATIONS",
            "Provide 5-7 actionable recommendations. For each: state the action, "
            "explain why it is recommended based on evidence, "
            "estimate effort (Low/Medium/High), and identify dependencies. "
            "Also list 3 future research directions.",
            analysis, entities)

        rich_print("  [dim]  → Conclusion...[/dim]")
        conclusion = self._section(
            "CONCLUSION",
            "Summarise the overall research outcome. Restate what was learned, "
            "what remains unsolved, and the significance of the findings.",
            analysis, entities)

        date_str = datetime.now().strftime("%B %d, %Y")
        header = (
            f"# Research Report: {entities.primary_topic}\n"
            f"**Domain:** {entities.domain}  \n"
            f"**Generated:** {date_str}  \n"
            f"**Research Theme:** {entities.research_theme}  \n\n"
            f"---\n\n"
        )
        return (header
                + f"# EXECUTIVE SUMMARY\n\n{summary}\n\n"
                + f"## RESEARCH PARAMETERS\n\n{params}\n\n"
                + f"## METHODOLOGY\n\n{methodology}\n\n"
                + f"## TECHNICAL ANALYSIS\n\n{tech_section}\n\n"
                + f"## RECOMMENDATIONS\n\n{recs}\n\n"
                + f"## CONCLUSION\n\n{conclusion}\n")


class MathTheoristAgent:
    """Derives a rigorous LaTeX mathematical framework adapted to the actual domain.
    Uses two passes: (1) identify the right framework, (2) write each section separately."""

    def __init__(self, client: OpenAI, model_name: str = "qwen3.5:9b"):
        self.client = client
        self.model = model_name

    def _identify_framework(self, entities: QueryEntities, analysis: str) -> dict:
        """First pass: decide WHAT mathematical framework fits this specific domain."""
        system = f"""
            You are a Mathematical Modelling Expert.
            Domain: {entities.domain}  |  Topic: {entities.primary_topic}

            Based on the technical analysis, identify the appropriate mathematical
            framework for formalising this research.

            Return JSON:
            {{
                "framework_name":        "<e.g. Convex Optimisation, Bayesian Inference, ...>",
                "primary_variables":     ["<var and its meaning>", ...],
                "core_equations_needed": ["<equation description>", ...],
                "relevant_theorems":     ["<theorem name>", ...],
                "notation_conventions":  "<LaTeX notation style>"
            }}

            CRITICAL: Match the framework to the ACTUAL domain ({entities.domain}).
            Do NOT default to signal processing if the domain is different.
        """
        raw = llm(self.client, self.model, system,
                  f"Analysis excerpt:\n\n{analysis[:3000]}", json_mode=True)
        return json.loads(raw)

    def _latex_section(self, section: str, instructions: str,
                        context: str, entities: QueryEntities, fw: dict) -> str:
        system = f"""
            You are a Professor writing the LaTeX "{section}" section.
            Domain: {entities.domain}  |  Topic: {entities.primary_topic}
            Framework: {fw['framework_name']}
            Variables:  {fw['primary_variables']}
            Notation:   {fw['notation_conventions']}

            Write ONLY the LaTeX body for this section
            (no \\documentclass, no \\begin{{document}}).
            {instructions}

            RULES:
            - Use \\begin{{equation}}...\\end{{equation}} for numbered equations.
            - Use \\begin{{align}}...\\end{{align}} for multi-line derivations.
            - Define every variable when first introduced.
            - Minimum 4 fully derived equations with explanations.
        """
        return llm(self.client, self.model, system,
                   f"Context:\n\n{context}", temperature=0.15)

    def derive_theory_latex(self, entities: QueryEntities, analysis: str) -> str:
        rich_print("  [dim]  → Identifying mathematical framework...[/dim]")
        fw = self._identify_framework(entities, analysis)
        rich_print(f"  [dim]     Framework: {fw['framework_name']}[/dim]")

        rich_print("  [dim]  → Problem Formulation...[/dim]")
        problem = self._latex_section(
            "Problem Formulation",
            "Define the problem formally: objective, constraints, decision variables. "
            "Introduce all notation.",
            analysis, entities, fw)

        rich_print("  [dim]  → System Model...[/dim]")
        model_sec = self._latex_section(
            "System Model",
            "Formally model the system or process from the analysis. "
            "Derive governing equations from first principles.",
            analysis, entities, fw)

        rich_print("  [dim]  → Proposed Solution...[/dim]")
        solution = self._latex_section(
            "Proposed Solution",
            "Derive the theoretical solution step-by-step. "
            "Show intermediate results. Prove key properties (convergence, optimality, etc.).",
            analysis, entities, fw)

        rich_print("  [dim]  → Complexity Analysis...[/dim]")
        complexity = self._latex_section(
            "Computational Complexity",
            f"Provide Big-O analysis. Relate to practical constraints of: "
            f"{entities.technologies_or_tools}. Compare with baseline methods.",
            analysis, entities, fw)

        date_str = datetime.now().strftime("%B %d, %Y")
        sub_topics_str = ", ".join(entities.sub_topics[:4])
        tools_str = ", ".join(entities.technologies_or_tools[:4]) or "general hardware"

        doc = (
            r"\documentclass{article}" "\n"
            r"\usepackage{amsmath, amssymb, amsthm, algorithm2e, hyperref, geometry}" "\n"
            r"\geometry{a4paper, margin=2.5cm}" "\n\n"
            f"\\title{{Theoretical Framework for {entities.primary_topic}}}\n"
            f"\\author{{Multi-Agent Research Pipeline \\\\ \\small Domain: {entities.domain}}}\n"
            f"\\date{{{date_str}}}\n\n"
            r"\newtheorem{theorem}{Theorem}" "\n"
            r"\newtheorem{lemma}{Lemma}" "\n"
            r"\newtheorem{definition}{Definition}" "\n"
            r"\newtheorem{proposition}{Proposition}" "\n\n"
            r"\begin{document}" "\n"
            r"\maketitle" "\n\n"
            r"\begin{abstract}" "\n"
            f"This document presents the formal mathematical framework for "
            f"\\textit{{{entities.primary_topic}}} within the domain of {entities.domain}. "
            f"The framework is based on {fw['framework_name']} and addresses "
            f"the following sub-topics: {sub_topics_str}. "
            f"Target implementation context: {tools_str}.\n"
            r"\end{abstract}" "\n\n"
            r"\section{Problem Formulation}" "\n"
            f"{problem}\n\n"
            r"\section{System Model}" "\n"
            f"{model_sec}\n\n"
            r"\section{Proposed Theoretical Solution}" "\n"
            f"{solution}\n\n"
            r"\section{Computational Complexity Analysis}" "\n"
            f"{complexity}\n\n"
            r"\end{document}"
        )
        return doc


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(user_query: str, model_name: str = "qwen3.5:9b"):
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    slug = make_slug(user_query)
    report_file = f"research_report_{slug}.md"
    tex_file    = f"theoretical_solution_{slug}.tex"

    rich_print(Rule("[bold cyan]Multi-Agent Research Pipeline[/bold cyan]"))
    rich_print(f"[bold cyan]Query:[/bold cyan] {user_query}")
    rich_print(f"[dim]Model:[/dim] {model_name}")
    rich_print(f"[dim]Outputs:[/dim] {report_file}  |  {tex_file}\n")

    # ── Step 1: Plan ──────────────────────────────────────────
    rich_print(Rule("[blue]Step 1 / 5 — Planning[/blue]"))
    planner = PlanningAgent(client, model_name)
    plan = planner.create_plan(user_query)
    rich_print(f"[green]Domain detected:[/green] {plan.domain}")
    rich_print(f"[green]Strategy:[/green] {plan.research_strategy}")
    rich_print(f"[green]Key challenges:[/green] {plan.key_challenges}")

    # ── Step 2: Entity extraction ─────────────────────────────
    rich_print(Rule("[blue]Step 2 / 5 — Entity Extraction[/blue]"))
    analyzer = QueryAnalysisAgent(client, model_name)
    entities = analyzer.analyze(user_query)
    rich_print(f"[green]Primary topic:[/green]  {entities.primary_topic}")
    rich_print(f"[green]Domain:[/green]         {entities.domain}")
    rich_print(f"[green]Sub-topics:[/green]     {entities.sub_topics}")
    rich_print(f"[green]Key concepts:[/green]   {entities.key_concepts}")
    rich_print(f"[green]Tools/Tech:[/green]     {entities.technologies_or_tools}")

    # ── Step 3: Data gathering ────────────────────────────────
    rich_print(Rule("[blue]Step 3 / 5 — Web Research[/blue]"))
    gatherer = DataGatherAgent(client, model_name)
    research_data = gatherer.gather_data(entities)
    n = research_data["result_count"]
    rich_print(f"[green]Collected {n} snippets from {len(research_data['search_queries'])} queries.[/green]")
    if n == 0:
        rich_print("[bold red]Warning: No web results. Analysis will rely solely on model knowledge.[/bold red]")

    # ── Step 4: Multi-pass analysis ───────────────────────────
    rich_print(Rule("[blue]Step 4 / 5 — Multi-Pass Analysis[/blue]"))
    analyst = AnalysisAgent(client, model_name)
    technical_analysis = analyst.analyze_research(entities, research_data)
    rich_print("[green]Analysis complete.[/green]")

    # ── Step 5a: Section-by-section report ───────────────────
    rich_print(Rule("[blue]Step 5a / 5 — Report Generation[/blue]"))
    reporter = ReportAgent(client, model_name)
    final_report = reporter.generate_final_report(plan, entities, technical_analysis)
    with open(report_file, "w") as f:
        f.write(final_report)
    rich_print(f"[bold green]Report saved:[/bold green] {report_file}")

    # ── Step 5b: LaTeX formalisation ─────────────────────────
    rich_print(Rule("[blue]Step 5b / 5 — Mathematical Formalisation[/blue]"))
    math_agent = MathTheoristAgent(client, model_name)
    latex_theory = math_agent.derive_theory_latex(entities, technical_analysis)
    with open(tex_file, "w") as f:
        f.write(latex_theory)
    rich_print(f"[bold green]LaTeX theory saved:[/bold green] {tex_file}")

    rich_print(Rule("[bold green]PIPELINE COMPLETE[/bold green]"))
    rich_print(f"  Report  → [cyan]{report_file}[/cyan]")
    rich_print(f"  Theory  → [cyan]{tex_file}[/cyan]\n")

    return report_file, tex_file


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Pipeline (local Ollama model)."
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Research query. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--model", default="qwen3.5:9b",
        help="Ollama model name (default: qwen3.5:9b).",
    )
    args = parser.parse_args()

    query = args.query or input("Enter your research query: ").strip()
    if not query:
        rich_print("[bold red]Error: No query provided.[/bold red]")
        sys.exit(1)

    run_pipeline(query, model_name=args.model)
