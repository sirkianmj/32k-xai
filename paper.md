---
title: '32k-XAI: A Mechanistic Simulation Platform for 32,000 Cognitive Agents'
tags:
  - Python
  - JAX
  - cognitive simulation
  - explainable AI
  - multi-agent reinforcement learning
authors:
  - name: Kian Mansouri Jamshidi
    affiliation: 1
  - name: Sabihe Khorshidi
    affiliation: 2
affiliations:
  - name: XAI Protocol Engineering, HPC
    index: 1
  - name: Psychometric Mapping, Empirical Validation
    index: 2
date: 25 October 2026
bibliography: paper.bib
---

# Summary

32k-XAI is a research platform for simulating a society of 32,000 autonomous cognitive agents. Each agent is governed by a transparent, mechanistically interpretable decision architecture parameterized by empirically grounded psychological variables. The platform enables causal, glass-box interrogation of emergent social phenomena by tracing macro-level outcomes to specific cognitive parameters at the individual level. No neural networks or large language models are used in the decision loop. Every decision can be traced to a named parameter with documented uncertainty and provenance.

# Statement of need

Contemporary multi-agent reinforcement learning (MARL) platforms rely on deep neural networks as opaque black boxes. This makes it impossible to causally attribute emergent social phenomena to specific cognitive mechanisms. 32k-XAI addresses this gap by providing a fully transparent, mechanistically interpretable alternative. It is designed for computational social scientists, psychologists, and XAI researchers who require both scalability (32,000 agents) and interpretability (per-decision traces).

# State of the field

Existing agent-based modeling platforms (e.g., Mesa, NetLogo) are interpretable but lack the psychometric grounding and scale of 32k-XAI. MARL platforms (e.g., PettingZoo) offer scale but are opaque. 32k-XAI uniquely combines a Vine Copula-based population generator with a JAX-vectorized decision engine that runs 32,000 agents in under 4 seconds on CPU.

# Software design

The platform separates population generation, decision-making, and interface. This modularity allows researchers to swap the copula, modify the decision loop, or extend the analysis dashboard without affecting the core engine. The deterministic run_id ensures reproducibility.

# Research impact statement

32k-XAI is currently used in ongoing research on the Dark Triad bifactor structure and DASS-21 distress modeling. Its five human benchmarks all pass within one standard error of published human data.

# AI usage disclosure

Generative AI was used to assist with code refactoring, documentation drafting, and workflow generation. All AI-generated content was reviewed, tested, and validated against the platform's human benchmarks and reproducibility criteria.

# Acknowledgements

We acknowledge the developers of JAX, FastAPI, and the open-source psychometric datasets that made this work possible.

# References
