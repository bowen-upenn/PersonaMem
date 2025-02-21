# Introduction

This folder has code to run several baseline approaches for the PersonaMem benchmark.

* Usage: `python run.py <top_k> <context_mode> <gpt_model>`
    * top_k: integer, how many candidate to retrieve in RAG
    * context_mode: choose between
      * `none`: question only
      * `rag`: use RAG, using BGE-M3 dense embedding for retrieval
      * `mem0`: use mem0 (this is slow, takes >25 min)
      * `all`: give the entire context history
    * gpt_model: which LLM model to use, currently support {"gpt-4o", "gpt-4o-mini"}

# Requirements:
* questions.csv and contexts.json must exist in ./data
* Packages:
    * numpy==2.2.2
    * pandas==2.2.3
    * FlagEmbedding==1.3.3
    * openai==1.61.1

# Results

| Model | Context | Performance |
| --- | --- | --- |
| gpt-4o | None | 0.76 |
| gpt-4o | RAG | 0.92 |
| gpt-4o-mini | None | 0.64 |
| gpt-4o-mini | RAG | 0.84 |
| gpt-4o-mini | mem0 | 0.84 |
| gpt-4o-mini | all | 0.8 |


![](bge-m3/result.png)


Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv. https://arxiv.org/abs/2402.03216
