# Introduction

This is a RAG baseline on the PersonaMem benchmark.
It uses BGE-M3 dense embedding for retrieval, and uses LLM for answering questions. 
* Usage: python run.py <top_k> <do_retrieval> <gpt_model>
* Requirement: questions.csv and contexts.json must exist in ./data

# Results

![](result.png)


Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv. https://arxiv.org/abs/2402.03216

