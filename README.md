# Fine-Tuning Qwen2-VL-7B-Instruct for Emotion Recognition in Autistic Children 🤖

## Overview
This project focuses on fine-tuning the **Qwen2-VL-7B-Instruct vision-language model** to automatically recognize facial expressions in autistic children. The model leverages both **visual and textual context** to classify emotions, providing a step toward AI-assisted emotion analysis for healthcare applications.

---

## Tech Stack🛠️
- **Language:** Python  
- **Libraries:** Transformers, BitsAndBytes, PEFT, TRL, PIL, Torch  
- **Frameworks:** PyTorch, LoRA fine-tuning, SFT Trainer  
- **Tools:** Kaggle for datasets, Jupyter Notebooks  
- **Hardware Optimization:** 4-bit quantization (NF4) for memory-efficient training  

---

## Dataset📂
- Source: Kaggle dataset of autistic children’s facial expressions  
- Format: JSON containing image paths and emotion labels  
- Emotions classified: `Natural`, `Anger`, `Fear`, `Joy`, `Sadness`, `Surprise`  🙂😡😨😊😢😮
- Preprocessing: Images resized to 224×224, text templates formatted for vision-language input  

---

## Model Architecture & Training🧩
- **Base Model:** `Qwen2VL-7B-Instruct`  
- **Fine-Tuning Technique:** LoRA (Low-Rank Adaptation)  
  - Targeted modules: `q_proj` and `v_proj`  
  - Hyperparameters: r=8, alpha=16, dropout=0.1  
- **Quantization:** 4-bit NF4 with double quantization for memory efficiency  
- **Training Configuration:**  
  - Epochs: 10  
  - Batch size: 1  
  - Learning rate: 2e-5  
  - Gradient accumulation: 4  
  - Max sequence length: 512  
  - Optimizer: Paged AdamW 32-bit  
- **Evaluation:** Small subset of training data used for evaluation  

---

## Approach📘
1. **Data Formatting:** Transform JSON samples into vision-language format for system-user-assistant interaction.  
2. **Image Preprocessing:** Convert images to RGB, resize, and filter invalid images.  
3. **Collation:** Batch images and texts using a custom collate function for the trainer.  
4. **Fine-Tuning:** Apply LoRA on Qwen2-VL-7B-Instruct with 4-bit quantization for GPU efficiency.  
5. **Text Generation & Evaluation:**  
   - Greedy decoding used for inference  
   - Model predicts only the emotion class corresponding to input images  

---

## Results📊
- Achieved stable **training and validation loss** (train: 3.14, validation: 2.56) under limited GPU constraints  
- Model effectively distinguishes between six emotion categories  
- Demonstrates **low-memory, efficient fine-tuning** of large vision-language models  

---
