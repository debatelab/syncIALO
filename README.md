# syncIALO 🤖🗯️

### What is this?

Synthetic drop-in replacements for _Kialo_ debate datasets which can be downloaded from the [kialo.com](https://kialo.com) debating website.

### Why?

The Kialo debates are a 👑 gold mine for NLP researchers, LLM developers and Critical Thinking scholars. Yet, the mine is ⛔️ barred (for them): Debate data downloaded or scraped from the website may not be used for research or commercial purposes in the absence of explicit permission or license agreement.

That's why the `Logikon AI` team has been creating this synthetic debate corpus which may serve as a drop-in replacement for the original Kialo data.

### Features

- permissive CC license
- reproducible and extendible
- open source code basis
- generated with open LLMs

### Corpora

| LLM🤖 | #debates[^1] | #claims[^1] | Notebook | Contributed by |
|---|---|---|---|---|
|[SOLAR](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)🤗|1000-50-50|583530-29722-29120|[link](src/solar-ipynb)|Logikon AI|

[^1]: Per train - eval - test split.

### Simulation Design

The following steps sketch the procedure by which debates are simulated:

1. Determine the debate's `tag cloud` by randomly sampling 8 topic tags.
2. Given the `tag cloud`, let 🤖 _generate_ a debate `topic` (e.g., a question).
3. Given the `topic`, let 🤖 _generate_ a suitable `motion` (e.g., the central claim).
4. Recursively generate an argument tree, starting with the `motion` as `target argument`:
   1. Let 🤖 _identify_ the implicit `premises` of the `target argument`.
   2. Let 🤖 _rank_ the implicit `premises` in terms of plausibility.
   3. Let 🤖 _generate_ k `pros` for the most plausible `premise`.
   4. Let 🤖 _generate_ k `cons` against the least plausible `premise`.
   5. Add `pros` and `cons` to argument tree, and use each of these as new `target argument` that is argued for and against, unless max depth has been reached.
