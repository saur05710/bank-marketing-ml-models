# Bank Marketing Term Deposit Classification Pipeline

## a. Problem Statement
Direct marketing campaigns (phone calls) of a Portuguese banking institution aim to predict whether a client will subscribe to a term deposit (variable `y`: `yes` or `no`). Accurately identifying potential subscribers enables the institution to optimize marketing resource allocation, focus outreach efforts on high-probability leads, and improve conversion rates while maintaining low contact overhead.

## b. Dataset Description
The dataset is derived from the UCI Bank Marketing repository (`https://archive.ics.uci.edu/dataset/222/bank+marketing`).Appropriately 20,000 entries was extracted and processed.

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
| Logistic Regression | 0.9560 | 0.9614 | 0.5464 | 0.2865 | 0.3759 | 0.3754 |
| Decision Tree | 0.9537 | 0.8868 | 0.5000 | 0.4108 | 0.4510 | 0.4294 |
| kNN | 0.9570 | 0.8642 | 0.5823 | 0.2486 | 0.3485 | 0.3623 |
| Naive Bayes | 0.9045 | 0.8833 | 0.2592 | 0.5730 | 0.3569 | 0.3421 |
| Random Forest (Ensemble) | 0.9573 | 0.9558 | 0.6750 | 0.1459 | 0.2400 | 0.3009 |

---

## Performance Observations & Evaluation

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the **highest AUC (0.9614)** across all models, demonstrating exceptional probability calibration and class-separation ability. Maintains a balanced trade-off between Precision (0.5464) and Recall (0.2865). |
| **Decision Tree** | Produced the **highest overall F1 Score (0.4510)** and **MCC (0.4294)**. It offers the most balanced performance for minority target prediction with a strong Recall of 0.4108 and Precision of 0.5000. |
| **kNN** | Delivered strong overall Accuracy (0.9570) and Precision (0.5823), but suffered from lower Recall (0.2486) as distance-based neighbors struggled with class imbalance. |
| **Naive Bayes** | Recorded the lowest Accuracy (0.9045) and Precision (0.2592), but achieved the **highest Recall (0.5730)** of all models due to feature independence assumptions catching positive cases at the expense of false positives. |
| **Random Forest (Ensemble)** | Yielded the **highest overall Accuracy (0.9573)** and **Precision (0.6750)** alongside an excellent AUC (0.9558), but produced the lowest Recall (0.1459) under default 0.5 classification thresholds due to heavy class imbalance. |
| **Overall Winner for your dataset?** | **Decision Tree** is the primary overall winner for imbalanced positive-class identification (leading with an F1 Score of 0.4510 and MCC of 0.4294). **Logistic Regression** serves as the optimal choice when prioritizing overall probability ranking and classification robustness (highest AUC of 0.9614). |
