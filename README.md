#  Algorithmic Justice League (AJL): Equitable AI for Dermatology

⚠️ _Remember to update the above title and remove all guidance notes and examples in this template, before finalizing your README_

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Haley Boswell | @Haley-B | -------|
| Jose Gonzalez | @josegonz115 | ---------|
| Harini Pootheri| @ |-------- |
| Hannah Sissoko| @ | ----------|

---

## 🎯 Project Highlights

### ✅ Best Model (Final Submission)
- Fine-tuned a Vision Transformer (ViT) using **transfer learning**, **class balancing**, and **skin tone-aware augmentation**
- Trained on a dataset of **~8000 images** combining internal Kaggle data with external sources
- Achieved **0.688 F1 score** on the final Kaggle leaderboard
- Ranked **7th overall** and **1st in the UCAL cohort**

---

### 📊 Datasets & Preprocessing

#### 🗂️ Internal Dataset (Kaggle)
- Original dataset with 6816 medical images across 21 skin condition classes  
- Applied image resizing, normalization, and augmentation via `Albumentations`

#### 🌐 External Dataset (Augmented)
- Combined HAM10000 + SD-198 with **standardized labels**  
- Merged using a custom mapping script, balanced to **500 samples per class**
- Augmentation included flips, rotation, elastic transforms, and light cropping

---

### 🧪 Model Experiments

#### 🧠 ViT Experiments
- Base ViT model trained on original Kaggle dataset  
- Best performance with learning rate `1e-5`, dropout `0.1`, and `7 epochs`

##### 🔁 ViT with External Dataset
- Trained with same hyperparameters on our **augmented dataset**
- Showed improved validation performance, but **worse Kaggle generalization**

#### 🧬 MedViT (Medical Vision Transformer)
- Tested MedViT with `MedViT_MedicalNet-22k` pretrained weights  
- Used full fine-tuning with dropout and cosine learning rate schedule  
- Model was very heavy and prone to **overfitting / memory issues**, yielding **low F1 (~0.33)**

---

### 🧰 Techniques & Tools
- Preprocessing: `Albumentations`, `OpenCV`, `PIL`, and `Pandas`
- Modeling: `PyTorch`, `HuggingFace Transformers`, `MedViT`
- Evaluation: `F1 Score (macro)`, accuracy, and Kaggle leaderboard ranking

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## **🏗️ Project Overview**

**Describe:**

* The Kaggle competition and its connection to the Break Through Tech AI Program
* The objective of the challenge
* The real-world significance of the problem and the potential impact of your work

---

## **📊 Data Exploration**

**Describe:**

* The dataset(s) used (i.e., the data provided in Kaggle \+ any additional sources)
* Data exploration and preprocessing approaches
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## **🧠 Model Development**

**Describe (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

**Answer the relevant questions below based on your competition:**

**WiDS challenge:**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?
2. How could your work help contribute to ADHD research and/or clinical care?

**AJL challenge:**

As Dr. Randi mentioned in her challenge overview, “Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”
As you answer the questions below, consider using not only text, but also illustrations, annotated visualizations, poetry, or other creative techniques to make your work accessible to a wider audience.
Check out [this guide](https://drive.google.com/file/d/1kYKaVNR\_l7Abx2kebs3AdDi6TlPviC3q/view) from the Algorithmic Justice League for inspiration!

1. What steps did you take to address [model fairness](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)? (e.g., leveraging data augmentation techniques to account for training dataset imbalances; using a validation set to assess model performance across different skin tones)
2. What broader impact could your work have?

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---
