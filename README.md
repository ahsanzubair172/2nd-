# 2nd-Body performance

# **Automated Machine Learning Classification with Hyperparameter Tuning**

## **Overview**

This project implements various machine learning classification algorithms with hyperparameter tuning to achieve optimal model performance. The dataset consists of biometric and physiological measurements to classify individuals into different categories. The focus is on evaluating models using different metrics, including accuracy, precision, recall, F1-score, and ROC-AUC.

## **Dataset**

The dataset consists of **13,393** records with **12** columns, including features such as age, gender, height, weight, body fat percentage, blood pressure readings, grip force, flexibility, sit-ups count, and broad jump distance. The target variable is the **class**, which is categorical.

### **Dataset Structure:**

| Column Name              | Data Type | Description                            |
| ------------------------ | --------- | -------------------------------------- |
| age                      | float64   | Age of the individual                  |
| gender                   | object    | Gender (categorical)                   |
| height\_cm               | float64   | Height in cm                           |
| weight\_kg               | float64   | Weight in kg                           |
| body fat\_%              | float64   | Body fat percentage                    |
| diastolic                | float64   | Diastolic blood pressure               |
| systolic                 | float64   | Systolic blood pressure                |
| gripForce                | float64   | Grip force measurement                 |
| sit and bend forward\_cm | float64   | Flexibility test result                |
| sit-ups counts           | float64   | Number of sit-ups performed            |
| broad jump\_cm           | float64   | Broad jump distance                    |
| class                    | object    | Target variable (classification label) |

## **Libraries Used**

The project utilizes the following Python libraries for data processing, model training, and evaluation:

- **Data Handling & Visualization**
  - `pandas`, `numpy` - Data manipulation and numerical operations
  - `matplotlib`, `seaborn` - Data visualization
- **Preprocessing & Model Selection**
  - `sklearn.preprocessing` (StandardScaler, LabelEncoder)
  - `sklearn.model_selection` (train\_test\_split, GridSearchCV, RandomizedSearchCV, cross\_val\_score)
- **Machine Learning Algorithms**
  - `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`
  - `LogisticRegression`, `SVC`, `KNeighborsClassifier`
  - `MLPClassifier` (Neural Network)
  - `XGBClassifier` (XGBoost)
  - `LGBMClassifier` (LightGBM)
- **Evaluation Metrics**
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `classification_report`, `confusion_matrix`, `roc_curve`
- **Warnings Handling**
  - `warnings.filterwarnings('ignore')` to suppress unnecessary warnings

## **Implementation Steps**

1. **Data Preprocessing**

   - Handle missing values (if any)
   - Encode categorical features using `LabelEncoder`
   - Standardize numerical features using `StandardScaler`
   - Split the dataset into **training (80%)** and **testing (20%)** sets

2. **Model Training and Hyperparameter Tuning**

   - Various classifiers are implemented with hyperparameter tuning using **GridSearchCV**
   - Example: Decision Tree tuning parameters:
     ```python
     param_grid = {
         'max_depth': [3, 5, 7, 10],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4]
     }
     grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
     grid_search.fit(X_train_scaled, y_train)
     ```

3. **Model Evaluation**

   - **Metrics Used:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - **Confusion Matrix and Classification Report:**
     ```python
     print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
     print("Classification Report:", classification_report(y_test, y_pred))
     ```
   - **ROC Curve Plotting for Multiclass Classification:**
     ```python
     fpr, tpr, _ = roc_curve(y_test, dt_pred_proba[:, 1], pos_label=1)
     plt.plot(fpr, tpr, label='ROC curve')
     ```

4. **Comparison of Models**

   - All classifiers are compared based on their evaluation scores
   - Best models are selected based on cross-validation scores

## **Results & Insights**

- The **Decision Tree Classifier** with hyperparameter tuning showed improved accuracy after optimizing `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **XGBoost and LightGBM** outperformed traditional algorithms due to their boosting capabilities.
- **MLPClassifier (Neural Network)** performed well but required additional tuning.
- Feature importance analysis helped in understanding the most significant predictors.

##

## **Installation & Usage**

### **Installation**

To install necessary dependencies, run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

For Kaggle Notebooks, install XGBoost and LightGBM using:

```bash
!pip install xgboost lightgbm --no-cache-dir
```

### **Run the Code**

1. Clone the repository (if applicable) or copy the code into a Jupyter Notebook/Kaggle Notebook.
2. Ensure all dependencies are installed.
3. Run the notebook cell-by-cell to preprocess data, train models, and evaluate results.

## **Conclusion**

This project demonstrates the power of **machine learning classification models** with **hyperparameter tuning** to enhance predictive accuracy. Through various **evaluation metrics**, we identified the most effective models for the given dataset. The insights gained can be used for further **AI-driven decision-making** in similar applications.

---

*Developed by Muhammad Ahsan Zubair*

