# syncIALO 🤖🗯️

### What is this?

**Syn**theti**c** drop-in replacements for [_K**ialo**_](https://kialo.com) debate datasets.

### Why?

The Kialo debates are a 👑 gold mine for NLP researchers, AI engineers, computational sociologists, and Critical Thinking scholars. Yet, the mine is legally ⛔️ barred (for them): Debate data downloaded or scraped from the website may not be used for research or commercial purposes in the absence of explicit permission or license agreement.

That's why the `DebateLab` team has built this python module for creating synthetic debate corpora, which may serve as a drop-in replacements for the Kialo data. We're synthesizing such data from scratch, simulating multi-agent debate and collaborative argument-mapping with 🤖 LLM-based agents. 

### Features

- permissive ODC license
- reproducible and extendable
- open source code basis
- works with open LLMs
- one-line-import as networkx graphs

### Corpora

| id | llm | # debates | ~# claims | link | contributed by |
|---|---|---|---|---|---|
| synthetic_corpus-001 |Llama-3.1-405B-Instruct|1000/50/50¹|560k/28k/28k¹|[HF hub→](https://huggingface.co/datasets/DebateLabKIT/syncialo-raw/viewer/synthetic_corpus-001)|DebateLab²|
| synthetic_corpus-001-DE |Llama-3.1-SauerkrautLM-70b-Instruct³|1000/50/50¹|560k/28k/28k¹|[HF hub→](https://huggingface.co/datasets/DebateLabKIT/syncialo-raw/viewer/synthetic_corpus-001-DE)|DebateLab|

¹ per train / eval / test split  
² with ❤️ generous support from 🤗 HuggingFace  
³ as translator


### Simulation Design

The following steps sketch the procedure by which debates are simulated:

1. Determine the debate's `tag cloud` by randomly sampling 8 topic tags.
2. Given the `tag cloud`, let 🤖 _generate_ a debate `topic` (e.g., a question).
3. Given the `topic`, let 🤖 _generate_ a suitable `motion` (i.e., the central claim).
4. Recursively generate an argument tree, starting with the `motion` as `target argument` ([code→](https://github.com/debatelab/syncIALO/blob/7db3b506271fe8a5c5d23c5c917635700c956516/src/syncialo/debate_builder.py#L340)):
   1. Let 🤖 _identify_ the implicit `premises` of the `target argument` ([code→](https://github.com/debatelab/syncIALO/blob/7db3b506271fe8a5c5d23c5c917635700c956516/src/syncialo/debate_builder.py#L91)).
   2. Let 🤖 _generate_ k `pros` for different `premises` of the `target argument` ([code→](https://github.com/debatelab/syncIALO/blob/7db3b506271fe8a5c5d23c5c917635700c956516/src/syncialo/chains/argumentation.py#L333)):
      * Choose `premise` to target in function of `premises`' plausibility.
      * Let 🤖 assume randomly sampled persona.
      * Generate 2k candidate arguments and select k most salient ones.
   3. Let 🤖 _generate_ k `cons` against different `premises` of the `target argument` ([code→](https://github.com/debatelab/syncIALO/blob/7db3b506271fe8a5c5d23c5c917635700c956516/src/syncialo/chains/argumentation.py#L444)):
      * Choose `premise` to target in function of `premises`' implausibility.
      * Let 🤖 assume randomly sampled persona.
      * Generate 2k candidate arguments and select k most salient ones.
   4. Check for and resolve duplicates via semantic similarity / vector store  ([code→](https://github.com/debatelab/syncIALO/blob/7db3b506271fe8a5c5d23c5c917635700c956516/src/syncialo/debate_builder.py#L244)).
   5. Add `pros` and `cons` to argument tree, and use each of these as new `target argument` that is argued for and against, unless max depth has been reached.

### Usage

Configure `workflows/synthetic_corpus_generation.py`. Then:

```sh
hatch shell
python workflows/synthetic_corpus_generation.py
```