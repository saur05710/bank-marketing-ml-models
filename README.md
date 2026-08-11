# File: project-folder/README.md

# Bank Marketing Term Deposit Classification Pipeline

## a. Problem Statement
Direct marketing campaigns (phone calls) of a Portuguese banking institution aim to predict whether a client will subscribe to a term deposit (variable `y`: `yes` or `no`). Accurately identifying potential subscribers enables the institution to optimize marketing resource allocation, focus outreach efforts on high-probability leads, and improve conversion rates while maintaining low contact overhead.

## b. Dataset Description
The dataset is derived from the UCI Bank Marketing repository (`https://archive.ics.uci.edu/dataset/222/bank+marketing`). Per requirements, a subset of exactly 20,000 entries was extracted and processed.

* **Total Records**: 20,000
* **Features**: 16 input features (6 numerical, 9 categorical, 1 target)
* **Numerical Attributes**: `age`, `day`, `duration` (last contact duration in seconds), `campaign` (number of contacts during current campaign), `pdays` (days passed after previous campaign contact), `previous` (number of contacts prior to campaign).
* **Categorical Attributes**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`.
* **Target Variable (`y`)**: Binary classification target indicating term deposit subscription (`1` for 'yes', `0` for 'no').

## c. GitHub Repository Link
[https://github.com/your-username/bank-marketing-ml-assignment](https://github.com/your-username/bank-marketing-ml-assignment)

---

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7895 | 0.8016 | 0.7371 | 0.4646 | 0.5700 | 0.4605 |
| Decision Tree | 0.7688 | 0.7650 | 0.6901 | 0.4172 | 0.5200 | 0.4005 |
| kNN | 0.7520 | 0.7142 | 0.6646 | 0.3514 | 0.4597 | 0.3453 |
| Naive Bayes | 0.7710 | 0.7689 | 0.6617 | 0.4854 | 0.5600 | 0.4192 |
| Random Forest (Ensemble) | 0.7833 | 0.7941 | 0.7694 | 0.3972 | 0.5239 | 0.4383 |

---

## Performance Observations & Evaluation

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates strong baseline performance with an Accuracy of 0.7895 and the highest overall AUC (0.8016). Linear decision boundaries effectively separate scaled features, maintaining a balanced F1 score (0.5700) and high MCC (0.4605). |
| **Decision Tree** | Captures non-linear feature interactions but exhibits moderate performance drop compared to ensemble methods due to decision boundary rigidity. Achieves 0.7688 Accuracy and 0.5200 F1 score. |
| **kNN** | Sensitive to high dimensionality and sparse one-hot encoded categorical variables. Recorded lower recall (0.3514) and the lowest MCC (0.3453) among evaluated algorithms. |
| **Naive Bayes** | High recall performance (0.4854) due to probabilistic independence assumption across features. Provides robust probabilistic baseline with 0.7710 Accuracy and 0.7689 AUC. |
| **Random Forest (Ensemble)** | Achieves the highest Precision (0.7694) among all models, drastically minimizing false positive subscription predictions. Overall robust Accuracy (0.7833) and AUC (0.7941). |
| **Overall Winner for your dataset?** | **Logistic Regression** is the primary overall winner due to the highest overall **AUC Score (0.8016)**, highest overall **Accuracy (0.7895)**, top **F1 Score (0.5700)**, and top **MCC Score (0.4605)**. However, **Random Forest** serves as the optimal production choice when precision is prioritized to minimize wasted marketing calls. |