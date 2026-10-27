# 🎓 Student Marks Prediction System

An end-to-end Machine Learning web application designed to predict student academic performance based on key lifestyle and study metrics. 

This project demonstrates practical data synthesis, feature engineering, data cleaning, and machine learning model training using Python's core data science stack.

---

## 📌 Project Overview

* **Backend & ML Pipeline (Built by Me):** Synthesized realistic custom datasets, handled data preprocessing/imputation, and trained a Random Forest Regression model.
* **Frontend UI (Created with AI assistance):** Built an interactive web dashboard using **Streamlit** to allow live user inputs and real-time score predictions.

---

## 📚 Learning Journey & Key Takeaways

Before coding this project, I first took the time to study the theoretical concepts behind the algorithms and data science tools, then applied them step-by-step:

1. **Lear Random Forest Regression:**
   - First, I learned the underlying theory of **Decision Trees** and how a **Random Forest Regressor** works as an ensemble method (combining multiple trees to improve accuracy and avoid overfitting).
   - Then, I learned how to implement it in Python using `scikit-learn`'s `RandomForestRegressor`, fitting the model on training data and making predictions on test sets.

2. **Data Creation & Synthesis (`NumPy`):** 
   - Used `np.random.randint()` and `np.arange()` to generate structured features (study hours, sleep hours, attendance, past performance).
   - Designed a weighted mathematical formula with Gaussian noise (`np.random.normal()`) to simulate realistic correlations between study habits and test scores.

3. **Data Manipulation & Cleaning (`Pandas`):**
   - Constructed DataFrames and managed duplicate entries using `.drop_duplicates()`.
   - Applied median imputation (`.fillna()`) across training sets to resolve potential missing values.

4. **Machine Learning Pipeline & Evaluation (`scikit-learn`):**
   - Split data into training and testing sets using `train_test_split()` (80/20 ratio).
   - Evaluated predictions using Mean Absolute Error (`mean_absolute_error`).

5. **App Deployment & UI Integration (`Streamlit`):**
   - Connected an ML backend model to interactive frontend sliders and metrics.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Data Synthesis & Analysis:** `numpy`, `pandas`
* **Machine Learning:** `scikit-learn` (Random Forest, Preprocessing, Metrics)
* **Frontend / UI Framework:** `streamlit`

---

## 📂 Project Structure

```text
Marks_prediction_system/
│
├── app.py                 # Core ML backend & Streamlit application
├── student_data.csv       # Synthetic dataset exported from Pandas
└── README.md              # Project documentation
📖 Installation & Developer Setup
Follow these steps to set up the environment and dependencies on your local machine:

1. Verify Python Environment
Ensure Python 3.10+ is installed on your system:

Bash
python --version
2. Install Required Packages
Open your terminal (in VS Code: Ctrl + ~ or Cmd + ~) and install all required libraries:

Bash
pip install pandas numpy scikit-learn streamlit
(Note: If you run into NumPy 2.0 version compatibility warnings, run pip install --upgrade numpy scipy scikit-learn streamlit).

⚙️ How to Run the Project
Option 1: Launch the Interactive Web App (Recommended)
Do NOT run Streamlit files using standard python app.py. Execute it using the Streamlit CLI:

Bash
streamlit run app.py
If streamlit is not recognized directly by your PATH, use the Python module runner:

Bash
python -m streamlit run app.py
# On macOS/Linux:
python3 -m streamlit run app.py
This automatically starts a local web server and opens the dashboard in your browser at http://localhost:8501.

Option 2: Run as standard Python Script (Data Generation/Export)
To run the underlying script to generate and export student_data.csv without launching the UI:

Bash
python app.py
# On macOS/Linux:
python3 app.py
