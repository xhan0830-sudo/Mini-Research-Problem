# Generative AI & LLMs: Fine-Tuning Transformer Decoders vs. Classical Encoders for Domain NLP

Mini Research Project (MRP) submission for the **Introductory Artificial Intelligence Course** (`fatheral/ai-intro-course`).

## 📌 Project Overview
This repository presents a comparative benchmark evaluating **Parameter-Efficient Fine-Tuning (PEFT)** via **Low-Rank Adaptation (LoRA)** on Causal Transformer Decoders (e.g., Llama-3-8B) against classical baselines (TF-IDF, Logistic Regression) and static Transformer Encoders (RoBERTa).

## 📁 Repository Structure
```text
ai-mrp-genai-llm/
├── README.md               # 10-minute presentation guide & setup instructions
├── requirements.txt        # Dependency specification
├── mrp_experiment.ipynb    # Interactive benchmark & visualization notebook
└── src/
    ├── train_lora.py       # Fine-tuning pipeline with AdamW and PEFT/LoRA
    └── evaluate.py         # Evaluation metrics (Accuracy, F1, Latency, VRAM)
