# 🖋️ Automatic Prompt Generation via Contrastive Self-Feedback

This repository provides an end-to-end pipeline for optimizing instructions in automated essay scoring systems using **Contrastive Self-Feedback**, leveraging Large Language Models (LLMs) to enhance evaluation accuracy.

---

## 🛠️ Installation

To get started, install the necessary dependencies using the following command:

"""$ pip install -r requirements.txt"""

---

## 🚀 Usage Guide

The pipeline consists of five main stages, from data preprocessing to final evaluation. Detailed configurations and execution commands for each step are provided below.

### 1. Data Preprocessing
Generate few-shot demonstration sets and evaluation datasets (1st and 2nd sets).

**Configuration:**
- path: Path to the raw dataset (e.g., ./data/path)
- few_shot_size: Number of few-shot sets to generate (Default: 20)
- eval_size: Number of evaluation samples (Default: 100)
- save_path: Destination path for the generated datasets (e.g., ./data/input/path)

"""$ python -m src.instruction_induction.data_preprocessing"""

---

### 2. Initial Instruction Pool Generation
Generate an initial pool of instructions by providing the few-shot demonstration sets to the LLaMA model.

**Configuration:**
- few_shot_sets_path: Path to the few-shot sets generated in Step 1
- ollama_model: LLM for instruction induction (e.g., llama3:70b-instruct)
- save_path: Path to save the initial instruction pool

"""$ python -m src.instruction_induction.induction"""

---

### 3. 1st Evaluation
Perform the first evaluation to measure the scoring accuracy (QWK) of the initial instruction pool.

**Configuration:**
- initial_instruction_path: Path to the initial instruction pool
- eval_path: Path to the 1st evaluation dataset
- ollama_model: LLM used for the scoring task (e.g., llama3:70b-instruct)
- save_path: Path to save the 1st evaluation results

"""$ python -m src.instruction_induction.eval"""

---

### 4. Contrastive Self-Feedback (CSF)
Compare the ground truth with model predictions to identify and autonomously revise low-quality instructions.

**Configuration:**
- initial_result_path: Path to the 1st evaluation results
- eval_path: Path to the evaluation dataset
- criteria_path: Path to the detailed evaluation criteria file (criteria.txt)
- ratio: Proportion of the bottom-tier instructions to be revised (e.g., 0.75)
- ollama_model: LLM for instruction revision
- save_path: Path to save the revised "Final Instruction Pool"

"""$ python -m src.contrasive_self_feedback.correction"""

---

### 5. 2nd Evaluation (Final)
Re-evaluate the revised instructions (Final Instruction Pool) on an independent dataset to verify performance improvements.

**Configuration:**
- final_instruction_path: Path to the revised instruction pool (CSF results)
- eval_path: Path to the 2nd evaluation dataset
- ollama_model: LLM used for final scoring
- save_path: Path to save the final evaluation results

"""$ python -m src.contrasive_self_feedback.eval"""

---

## 📊 Evaluation Metric

This project utilizes the **Quadratic Weighted Kappa (QWK)** metric to assess the consistency and agreement between the model's predicted scores and the human-assigned ground truth.

---

