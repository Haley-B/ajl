#  Algorithmic Justice League (AJL): Equitable AI for Dermatology

⚠️ _Remember to update the above title and remove all guidance notes and examples in this template, before finalizing your README_

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Haley Boswell | @Haley-B | Led end-to-end development including exploratory data analysis (EDA), dataset curation (5 custom sets), label standardization, augmentation experiments, and submission strategy. Trained and evaluated ViT-Base, ViT-Large, ViT-Large (ImageNet-21k), MedViT-Base, and MedViT-Large (MedicalNet-22k) models. Maintained all notebooks and code infrastructure. |
| Jose Gonzalez | @josegonz115 | Contributed to EDA and created insightful visualizations. Trained and evaluated a ResNet50 model as part of model comparison. |
| Harini Pootheri| @ | Provided consistent team support through active participation and collaboration. Helped foster a positive and communicative team dynamic. |
| Hannah Sissoko| @ | No contribution after initial team formation. |

---

# **🏗️ Project Overview**
**The Kaggle Competition and Its Connection to the Break Through Tech AI Program**

The Kaggle competition was an integral component of the Break Through Tech AI Program, designed to bridge the gender gap in artificial intelligence by providing women and underrepresented groups with real-world AI challenges. Participants were tasked with developing machine learning models to address specific problems, fostering practical experience and innovation.

**Objective of the Challenge**

The primary objective of this challenge was to develop an AI-driven model capable of accurately classifying different skin conditions across diverse skin tones. This entailed creating a robust dataset representative of various skin types and training models to ensure equitable diagnostic performance, regardless of a patient's skin color.

**Real-World Significance and Potential Impact**

The underrepresentation of darker skin tones in medical imagery has profound implications on healthcare outcomes. A 2020 study revealed that only 18% of images in dermatology textbooks depicted dark skin tones, underscoring a significant gap in medical education. This lack of representation can lead to misdiagnoses or delayed diagnoses for patients with darker skin, further exacerbating health disparities.

Other studies have found similarly concerning statistics:
  - A 2018 study reported that only 4.5% of images in medical textbooks featured darker skin tones.
  - A review of 15 nursing textbooks found that only 12.3% of photo images and 2.4% of drawn graphics represented dark skin tones.
  - Another study analyzing 1,123 dermatology teaching images showed that just 14.9% featured skin of color (SoC), while 84.3% featured lighter tones.
  - Skin conditions like eczema and alopecia, which are prevalent in SoC populations, were among the least represented.

These disparities can have life-altering consequences. For example, research shows that melanoma—among the most deadly skin cancers—is diagnosed later and more aggressively in patients with darker skin due to lack of diagnostic familiarity. In one documented case, a biracial patient went undiagnosed with T-cell lymphoma for five years, being repeatedly misdiagnosed due to lack of awareness of how the condition appears on darker skin.

In parallel, generative AI and diagnostic models trained disproportionately on light-skinned imagery may fail to recognize conditions in people with SoC, compounding existing inequalities in healthcare access and outcomes.

By incorporating skin tone-aware augmentation, balanced class sampling, and diverse datasets, our project directly addresses these shortcomings. Our model development process prioritized fairness and inclusivity, creating AI that does not just perform well but performs equitably.

Efforts like this are a step toward closing the diagnostic gap, enhancing medical education, and building AI systems that work for everyone—regardless of skin tone. As AI continues to expand its footprint in clinical tools, representation is not just ethical—it’s **lifesaving**.
---

# **🎯 Project Highlights**

### ✅ **Best Model (Final Submission)**

- Fine-tuned a **Vision Transformer** model (**ViT-Large-Patch16-224**, pretrained on **ImageNet-21k**) using **transfer learning**, **targeted class balancing**, and **skin tone-aware augmentation**.
- Trained on a dataset of **4,270 images**, with **3,416 used for training** and **854 for validation**.
- This dataset included **untouched** and **minimally augmented** images from the original **Kaggle source data**.
  - Augmentations applied: **horizontal flip**, **vertical flip**, and **rotation** between **-45° to +45°**.
- Balanced skin tone representation using both **Fitzpatrick Centaur (FC)** and **Fitzpatrick Scale (FS)** labels:
  - Created skin tone subsets where **FC = FS**, then **augmented all groups** to match the largest (442 images).
  - Remaining images (where **FC ≠ FS**) were included to preserve data diversity.
- Conducted a **hyperparameter grid search** to optimize performance.
  - Final settings: **learning rate = 2e-5**, **weight decay = 0.01**.
- Trained for **5 epochs** to allow learning while minimizing **overfitting**.
- Used **AdamW** optimizer and **cross-entropy loss** from **PyTorch**.
- Achieved a final **macro F1 score of 0.66826** on the **Kaggle Private Leaderboard**.
- Ranked **7th overall** and secured **1st place in the UCLA cohort**.

#### 🧰 Techniques & Tools
- Preprocessing: `Albumentations`, `OpenCV`, `PIL`, and `Pandas`
- Modeling: `PyTorch`, `HuggingFace Transformers`, `MedViT`
- Evaluation: `F1 Score (macro)`, accuracy, and Kaggle leaderboard ranking

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

# **📊 Data Exploration**

#### 🗂️ Internal Dataset (Kaggle)
- Subset of the FitzPatrick17k dataset with 2860 medical images across 21 skin condition classes. 
- Applied image resizing, normalization, and augmentation via `Albumentations`.

#### 🌐 External Dataset (Augmented)
- Combined HAM10000 + SD-198 + PAD-UFES-20 + ASCID with **standardized labels**.  
- Merged using a custom mapping script, balanced to **500 samples per class**.
- Augmentation included flips, rotation, elastic transforms, and light cropping.

## 🗃️ Source Data Overview

### Columns

Our dataset consisted of **2,860 rows** and **8 columns**, described in the table below:

| Column | Name                  | Description                                                                                          | Used? |
|--------|-----------------------|------------------------------------------------------------------------------------------------------|-------|
| 1      | `md5hash`             | Image Identifier ID / File Name                                                                      | Yes |
| 2      | `fitzpatrick_scale`   | Integer values -1, 1–6 describing **self-identified Fitzpatrick skin type**                         | Yes |
| 3      | `fitzpatrick_centaur`| Integer values -1, 1–6 describing **Centaur Labs-assigned Fitzpatrick skin type**                   | Yes |
| 4      | `label`               | The medical diagnosis label (e.g., eczema, melanoma, etc.)                                          | Yes |
| 5      | `nine_partition_label`| Value indicating one of nine diagnostic categories                                                  | No |
| 6      | `three_partition_label`| Value indicating one of three diagnostic categories                                                | No |
| 7      | `qc`                  | Quality control check by a **Board-certified dermatologist** (only available for limited images)    | No |
| 8      | `ddi_scale`           | Used to reconcile this dataset with another — **not relevant for our use case**                    | No |

### Labels/Target

As mentioned, our labels are meant to classify various skin conditions. Shown below are the various coniditions and their distribution among the dataset:
![Skin Condition Distribution](images/skin_dist.png)

### Fitzpatrick Values

The key challenge in this project was handling the imabalance among skin tones, which was quite severe for darker skin tones (those with values >= 4 for FC/FS). We see this imbalance below:
![Fitzpatrick Skin Type Distribution](images/fitz_skintype_dist.png)




* The dataset(s) used (i.e., the data provided in Kaggle \+ any additional sources)
* Data exploration and preprocessing approaches
* Challenges and assumptions when working with the dataset(s)



## Data Augmentation and Expansion Via External Datasets

We ultimately created 5 unique datasets using various methods, including augmentation and sourcing external data.

### Augmentations Applied:

- Horizontal Flips, Vertical Flips, Rotations (-45 to +45 degrees), Elastic Transforms, Grid Distortions, Affine Transformations (for image shifts only), Mild Cropping, Transpose, Mild Perspective Shifts

### External Datasets Used:

| Name | Description                  | Relevancy                                                                                       | Images Gained |
|--------|-----------------------|------------------------------------------------------------------------------------------------------|-------|
| HAM10000     | `md5hash`             | Image Identifier ID / File Name                                                                      | Yes |
| SD-198     | `fitzpatrick_scale`   | Integer values -1, 1–6 describing **self-identified Fitzpatrick skin type**                         | Yes |
| PAD-UFES-20     | `fitzpatrick_centaur`| Integer values -1, 1–6 describing **Centaur Labs-assigned Fitzpatrick skin type**                   | Yes |
| ASCID    | `label`               | The medical diagnosis label (e.g., eczema, melanoma, etc.)                                          | Yes |
 

#### Dataset 1: 
  - Size
  - Extneral Images Used?
  - Augmentations Applied
  - 

#### Dataset 2:

#### Dataset 3:

#### Dataset 4:

#### Dataset 5:

---

# **👩🏽‍💻 Setup & Execution**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

# **🧠 Model Development**

### 🧪 Model Experiments

#### 🧠 ViT Experiments
- Base ViT model trained on original Kaggle dataset  
- Best performance with learning rate `1e-5`, dropout `0.1`, and `7 epochs`

#### 🔁 ViT with External Dataset
- Trained with same hyperparameters on our **augmented dataset**
- Showed improved validation performance, but **worse Kaggle generalization**

#### 🧬 MedViT (Medical Vision Transformer)
- Tested MedViT with `MedViT_MedicalNet-22k` pretrained weights  
- Used full fine-tuning with dropout and cosine learning rate schedule  
- Model was very heavy and prone to **overfitting / memory issues**, yielding **low F1 (~0.33)*
  
---

**Describe (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

# **📈 Results & Key Findings**

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
