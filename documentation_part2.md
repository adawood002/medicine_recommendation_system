## CHAPTER 4 — SYSTEM DESIGN

### 4.1 System Architecture
The proposed system follows a modular architecture consisting of three primary layers: the Data Layer, the Logic Layer, and the Presentation Layer. 
- **Data Layer:** Contains the CSV datasets and the serialized model artifacts (.pkl and .keras files).
- **Logic Layer:** This is the core of the system, written in Python. it handles data pre-processing, model inference (prediction), and the medicine ranking algorithm.
- **Presentation Layer:** Built using Streamlit, this layer manages user inputs and displays the final recommendations through a web browser.

### 4.2 Data Flow Diagram (DFD)
- **Level 0 (Context Diagram):** The user provides symptoms and patient profile to the "Medicine Recommendation System," which outputs a predicted disease and medicine recommendations.
- **Level 1 DFD:**
  1. **User Input Process:** Collects symptoms and demographics.
  2. **Feature Engineering Process:** Scales and encodes the input.
  3. **Inference Process:** Uses the trained ML/DL models to predict the disease.
  4. **Recommendation Process:** Maps disease to medicine database and applies ranking.
  5. **Display Process:** Renders the results on the UI.

### 4.3 Use Case Diagram
- **Actor:** Patient/User
- **Use Cases:**
  - View Home Page
  - Input Symptoms and Personal Details
  - Generate Diagnosis and Recommendation
  - View Model Performance Metrics
  - Download Recommendation Report

### 4.4 Sequence Diagram
1. User interacts with the Streamlit interface.
2. Streamlit sends input data to the Recommendation Engine.
3. The Engine calls the Pre-processing module to scale data.
4. The Engine loads the saved Model and gets a prediction.
5. The Engine queries the Medicine Database.
6. The Engine applies the Ranking Algorithm.
7. Results are returned to Streamlit and displayed to the User.

### 4.5 Entity Relationship (ER) Diagram
The system uses a relational logic even though data is stored in CSVs:
- **Disease Entity:** Primary Key (DiseaseID), Name, Description.
- **Symptom Entity:** Name, Type.
- **Medicine Entity:** Name, Dosage, Warnings, SafetyProfile, Effectiveness.
- **Mapping:** Disease has a One-to-Many relationship with Medicines.

### 4.6 Module Interaction
The modules are loosely coupled. The `train.py` script generates artifacts that `recommendation.py` uses. The `app.py` script serves as the orchestrator that brings all modules together into a unified user experience.

---

## CHAPTER 5 — SDLC MODEL

### 5.1 Chosen SDLC Model: Agile Methodology
The Agile Methodology was chosen for this project due to its iterative nature and flexibility. Healthcare projects often require adjustments based on model performance results, and Agile allows for continuous integration and improvement.

### 5.2 Agile Sprints Breakdown
- **Sprint 1 (Weeks 1-2):** Requirement analysis and data collection. Generation of synthetic symptoms and medicine datasets.
- **Sprint 2 (Weeks 3-4):** Implementation of data pre-processing pipeline and initial ML model training.
- **Sprint 3 (Weeks 5-6):** Deep learning model development (ANN/LSTM) and initial evaluation.
- **Sprint 4 (Weeks 7-8):** Hyperparameter tuning and development of the medicine ranking engine.
- **Sprint 5 (Weeks 9-10):** Streamlit UI development and integration with the backend engine.
- **Sprint 6 (Weeks 11-12):** Final testing, bug fixing, and documentation preparation.

### 5.3 Gantt Chart (Project Timeline)

| Activity | W1-2 | W3-4 | W5-6 | W7-8 | W9-10 | W11-12 |
|---|---|---|---|---|---|---|
| Data Collection | ██ | | | | | |
| ML Training | | ██ | | | | |
| DL Training | | | ██ | | | |
| Optimization | | | | ██ | | |
| UI Development | | | | | ██ | |
| Testing/Docs | | | | | | ██ |

### 5.4 Risk Analysis

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| Low Model Accuracy | Medium | High | Use Ensemble methods and deep tuning |
| Overfitting | High | Medium | Apply Dropout and K-Fold CV |
| Invalid User Input | Low | Medium | Implement strict input validation in UI |
| Dataset Bias | Medium | High | Use stratified splitting and varied synthetic data |

### 5.5 Why Agile over Waterfall?
The Waterfall model is too rigid for machine learning projects where the outcome of training often dictates the next steps. Agile allowed us to refine our feature selection and ranking logic iteratively as we discovered how different models responded to the data.

---

## CHAPTER 6 — IMPLEMENTATION

### 6.1 Development Environment Setup
- **IDE:** VS Code with Python extensions.
- **Environment:** A virtual environment was created to manage dependencies listed in `requirements.txt`.
- **Training:** Google Colab was used for training deep learning models to leverage GPU acceleration.

### 6.2 Dataset Description & Statistics
- Total Samples: 3,000
- Total Features: 62 (Symptoms + Age + Gender)
- Classes: 20 Diseases
- Medicine Database: 65 Unique Medications

### 6.3 Pre-Processing Implementation Details
The pre-processing module (`preprocess.py`) handles duplicate removal and feature scaling. We utilized the `StandardScaler` from Scikit-learn to ensure that the 'Age' feature (range 1-100) does not dominate the binary symptom features (0-1).

### 6.4 Model Training Code Walkthrough
The `train.py` script implements a loop that iterates through various classifiers. Each model is fitted on `X_train`, and its performance is validated on `X_test`. For the ANN, a sequential model with three hidden layers and dropout was implemented to capture non-linear symptom correlations.

### 6.5 Recommendation Engine Logic
The core logic in `recommendation.py` uses the `predict_proba()` method. This allows the system to provide a confidence score. If the confidence is below a certain threshold, the system displays a warning. The medicine ranking uses a custom scoring function that adjusts medicine 'effectiveness' based on patient 'existing conditions'.

### 6.6 Streamlit Application Structure
The application uses a multi-page sidebar layout. Streamlit's `@st.cache_resource` is used to load the ML models into memory only once, ensuring the app remains fast and responsive during user sessions.

### 6.7 Challenges Faced & Solutions
- **Challenge:** Symptom overlap between similar diseases (e.g., Flu and COVID-19).
- **Solution:** Integrated deep learning models (ANN) which provided better class separation than simple Naive Bayes.
- **Challenge:** Handling missing symptoms.
- **Solution:** Defaulted missing symptoms to '0' (absent) during inference to maintain feature vector consistency.

---

*Summary: This section detailed the design, methodology, and implementation specifics of the system. The use of Agile and a modular architecture ensured a robust and flexible development process.*
