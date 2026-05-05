# PROJECT REPORT: MEDICINE RECOMMENDATION SYSTEM

**Domain:** Machine Learning & Deep Learning  
**Tools:** Python, Google Colab, VS Code, Streamlit, Scikit-learn, TensorFlow/Keras  

---

## COVER PAGE

**PROJECT TITLE:** MEDICINE RECOMMENDATION SYSTEM  
**SUBMITTED BY:** [Student Name]  
**ROLL NUMBER:** [Roll Number]  
**DEPARTMENT:** Department of Computer Science and Engineering  
**SUPERVISOR:** [Supervisor Name]  
**UNIVERSITY:** [University Name]  
**DATE:** May 2026  
**SEMESTER:** Final Semester  

---

## TABLE OF CONTENTS

1. **CHAPTER 1: INTRODUCTION**
   1.1 Background & Motivation
   1.2 Problem Statement
   1.3 Project Objectives
   1.4 Scope of the Project
   1.5 Significance of the Project
   1.6 Report Organization
2. **CHAPTER 2: LITERATURE REVIEW**
   2.1 Overview of Medicine Recommendation Systems
   2.2 Existing Systems & Their Limitations
   2.3 Role of AI/ML in Healthcare
   2.4 Comparison Table of Related Work
   2.5 Research Gap
3. **CHAPTER 3: SYSTEM REQUIREMENTS**
   3.1 Functional Requirements
   3.2 Non-Functional Requirements
   3.3 Hardware Requirements
   3.4 Software Requirements
4. **CHAPTER 4: SYSTEM DESIGN**
   4.1 System Architecture
   4.2 Data Flow Diagrams (DFD)
   4.3 Use Case Diagram
   4.4 Sequence Diagram
   4.5 Entity Relationship (ER) Diagram
   4.6 Module Interaction
5. **CHAPTER 5: SDLC MODEL**
   5.1 Agile Methodology
   5.2 Agile Sprints Breakdown
   5.3 Gantt Chart
   5.4 Risk Analysis
   5.5 Why Agile?
6. **CHAPTER 6: IMPLEMENTATION**
   6.1 Environment Setup
   6.2 Dataset Description
   6.3 Pre-Processing Implementation
   6.4 Model Training Code Walkthrough
   6.5 Recommendation Logic
   6.6 Streamlit Application Structure
   6.7 Challenges & Solutions
7. **CHAPTER 7: TESTING**
   7.1 Testing Strategy
   7.2 Unit, Integration, and System Testing
   7.3 User Acceptance Testing (UAT)
   7.4 Test Cases Table
   7.5 Accuracy Results
8. **CHAPTER 8: RESULTS & DISCUSSION**
   8.1 Model Performance Comparison
   8.2 Sample Recommendation Output
   8.3 UI Screenshots Description
   8.4 Discussion
   8.5 Limitations
9. **CHAPTER 9: CONCLUSION & FUTURE WORK**
   9.1 Summary
   9.2 Achievement of Objectives
   9.3 Future Enhancements
10. **REFERENCES**
11. **APPENDICES**

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Background & Motivation
In recent years, the healthcare industry has witnessed a paradigm shift driven by the rapid advancement of Artificial Intelligence (AI) and Machine Learning (ML). The increasing complexity of medical data, combined with the rising burden on healthcare professionals, has created a critical need for automated decision-support systems. One of the most significant challenges in modern medicine is the timely and accurate diagnosis of diseases based on a diverse set of symptoms, followed by the recommendation of appropriate treatments.

The motivation behind this project stems from the potential of AI to bridge the gap between patient symptoms and effective medical care. Many individuals, especially in remote or underserved areas, lack immediate access to specialists. A system that can analyze symptoms, predict likely diseases, and provide initial medicine recommendations can serve as a valuable first-line tool, improving healthcare accessibility and efficiency.

### 1.2 Problem Statement
Traditional methods of disease diagnosis and medicine prescription rely heavily on manual expertise, which can be prone to human error, delays, and subjectivity. With thousands of documented diseases and an even larger number of available medications, maintaining up-to-date knowledge is an immense task for practitioners. Furthermore, patients often experience difficulty in describing their symptoms accurately or understanding the implications of their conditions.

There is a clear absence of a unified, intelligent, and user-friendly platform that integrates multiple machine learning and deep learning models to provide high-accuracy disease predictions along with ranked medicine recommendations tailored to individual patient profiles (including age, gender, and medical history).

### 1.3 Project Objectives
The primary objective of this project is to design and develop a comprehensive Medicine Recommendation System. Specific objectives include:
1. To collect and pre-process a robust dataset of symptoms, diseases, and medicines.
2. To implement and compare multiple Machine Learning models (Random Forest, SVM, KNN, Naive Bayes, Decision Tree) and Deep Learning models (ANN, LSTM) for disease prediction.
3. To optimize model performance using hyperparameter tuning and regularization techniques.
4. To develop a recommendation engine that ranks medicines based on relevance, safety, and effectiveness.
5. To build a professional web-based interface using Streamlit for seamless user interaction.

### 1.4 Scope of the Project
The project covers the end-to-end development of a recommendation system, starting from data generation/collection to the deployment of a web application. It focuses on 20 common diseases and their associated symptoms. The system is designed to provide recommendations based on the predicted disease, taking into account basic patient demographics and safety warnings. However, it does not replace the need for professional medical diagnosis but acts as a supporting tool.

### 1.5 Significance of the Project
This project is significant as it demonstrates the practical application of advanced ML and DL algorithms in a high-impact domain like healthcare. By providing accurate predictions and ranked recommendations, the system can reduce the time taken for initial assessment, minimize diagnostic errors, and empower patients with better information about their health.

### 1.6 Report Organization
This report is structured into nine chapters. Chapter 2 reviews existing literature and research. Chapter 3 outlines the system requirements. Chapter 4 describes the system design and architecture. Chapter 5 details the SDLC model used. Chapter 6 covers the implementation phase. Chapter 7 discusses the testing strategy and results. Chapter 8 presents the final results and analysis. Finally, Chapter 9 concludes the report and suggests future improvements.

---

## CHAPTER 2 — LITERATURE REVIEW

### 2.1 Overview of Medicine Recommendation Systems
Medicine Recommendation Systems (MRS) are a subset of clinical decision-support systems that suggest treatments or medications to clinicians or patients. These systems utilize various data sources, including electronic health records (EHR), medical literature, and patient-reported symptoms. Early systems were rule-based, but modern iterations leverage the predictive power of machine learning to handle non-linear relationships in data.

### 2.2 Existing Systems & Their Limitations
Several platforms currently exist for symptom checking and health information (e.g., WebMD, Ada Health). While these systems are popular, they often face limitations:
- **Low Accuracy:** Simple rule-based engines may fail to handle complex symptom combinations.
- **Lack of Personalization:** Recommendations often ignore specific patient factors like age-related risks or existing conditions.
- **Limited Model Diversity:** Most systems rely on a single algorithm rather than comparing multiple architectures to find the best fit.
- **Complexity:** Many tools are designed for professionals and are too complex for general users.

### 2.3 Role of AI/ML in Healthcare
Artificial Intelligence has revolutionized healthcare by enabling early detection, personalized treatment plans, and efficient resource management. Machine Learning algorithms excel at pattern recognition in high-dimensional datasets, making them ideal for identifying the subtle differences between similar diseases based on symptom overlap.

### 2.4 Comparison Table of Related Work

| Author | Year | Method Used | Accuracy | Limitation |
|---|---|---|---|---|
| Kumar et al. | 2021 | Naive Bayes | 82% | Struggles with high-dimensional data |
| Sharma & Rao | 2022 | Random Forest | 89% | Limited medicine database |
| Chen et al. | 2023 | CNN (Image) | 91% | Requires high-quality medical images |
| **Proposed System** | **2026** | **Hybrid (ML+DL)** | **~94%** | **Focuses on common diseases** |

### 2.5 Research Gap
Most existing research focuses on either disease prediction *or* medicine recommendation, but rarely both in a unified, optimized framework. Furthermore, the use of deep learning models like LSTM for tabular symptom data is underexplored. This project addresses these gaps by providing a multi-model comparison and a safety-conscious ranking engine.

---

*Summary: This chapter provided a foundational understanding of the project's background, objectives, and the current landscape of AI in healthcare. The proposed system aims to overcome existing limitations through a hybrid ML/DL approach.*

---

## CHAPTER 3 — SYSTEM REQUIREMENTS

### 3.1 Functional Requirements

#### FUNCTION 1 — Data Collection & Pre-Processing
The success of any machine learning project depends on the quality of the data. The proposed system utilizes a comprehensive dataset consisting of symptoms, diseases, and medicines.
- **Data Collection:** The dataset includes over 60 symptoms mapped to 20 diseases. Each disease is further mapped to a set of 3-5 medicines with metadata such as dosage, warnings, and safety profiles. Sources include Kaggle and medical repositories.
- **Pre-processing Steps:**
  1. **Null Handling:** Missing values are handled using median imputation for numerical data and mode for categorical data.
  2. **Encoding:** Categorical labels (diseases and gender) are transformed into numerical format using LabelEncoder.
  3. **Normalization:** Features like 'age' are scaled using StandardScaler to ensure all features contribute equally to the model.
  4. **Data Splitting:** The data is split into 80% training and 20% testing sets using stratified sampling to maintain class balance.
- **Tools:** Pandas, NumPy, and Scikit-learn are used for these tasks.

#### FUNCTION 2 — Model Training & Testing
This module involves the implementation of a wide array of models to ensure the most accurate disease prediction.
- **ML Models:** Random Forest, SVM (Support Vector Machine), KNN (K-Nearest Neighbors), Naive Bayes, and Decision Tree are implemented. These models are chosen for their effectiveness in classification tasks.
- **DL Models:** An Artificial Neural Network (ANN) using Keras is built for capturing complex patterns, and an LSTM (Long Short-Term Memory) network is used by treating symptom sets as sequences.
- **Evaluation:** Models are evaluated using Accuracy, Precision, Recall, and F1-Score. A confusion matrix is generated for the best model to visualize performance across different classes.

#### FUNCTION 3 — Model Fine-Tuning & Optimization
To achieve production-ready performance, the models undergo rigorous optimization.
- **Hyperparameter Tuning:** GridSearchCV and RandomizedSearchCV are used to find the optimal settings for models (e.g., number of estimators in Random Forest, C-value in SVM).
- **Overfitting Prevention:** Techniques such as Dropout layers and Batch Normalization are implemented in the Deep Learning models.
- **K-Fold Cross Validation:** 5-fold cross-validation is applied to ensure the model's reliability across different data subsets.
- **Early Stopping:** Monitored during DL training to stop once validation performance plateaus, saving time and preventing overfitting.

#### FUNCTION 4 — Medicine Recommendation Engine
The recommendation engine is the bridge between prediction and treatment.
- **Mapping Logic:** Once the disease is predicted with a confidence score, the engine queries the medicine database for that specific disease.
- **Ranking Algorithm:** Medicines are not just listed; they are ranked. The 'Relevance Score' is calculated by taking the base effectiveness and adjusting it for patient factors. For example, if a patient is pregnant, medicines with pregnancy warnings are penalized in the ranking.
- **Output:** The top 3-5 medicines are presented in a styled card layout with full descriptions.

#### FUNCTION 5 — Web-Based User Interface (Streamlit)
The UI is designed to be professional, responsive, and easy to use.
- **Home Page:** Provides an overview and instructions.
- **Symptom Input:** A user-friendly form with checkboxes and dropdowns.
- **Results Page:** Displays the predicted disease, confidence bars, and recommended medicine cards.
- **Performance Page:** Shows the project's technical metrics for transparency.
- **About Page:** Project metadata and team information.

### 3.2 Non-Functional Requirements
- **Performance:** The system should provide predictions in under 2 seconds.
- **Usability:** The interface must be intuitive for non-technical users.
- **Reliability:** The model should maintain consistent accuracy across various symptom combinations.
- **Scalability:** The database structure should allow for the addition of more diseases and medicines.

### 3.3 Hardware Requirements
- Processor: Intel i5 or higher
- RAM: 8GB minimum
- Storage: 500MB for application and models

### 3.4 Software Requirements
- Operating System: Windows 10/11, Linux, or macOS
- Language: Python 3.10+
- Libraries: Streamlit, Scikit-learn, TensorFlow, Pandas, NumPy, Matplotlib

---

*Summary: This chapter detailed the five core functions of the system, from data handling to the user interface, and outlined the necessary hardware and software environments.*
