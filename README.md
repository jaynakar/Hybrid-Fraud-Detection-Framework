## 🧠 Hybrid Fraud Detection Framework

A Hybrid and Explainable Fraud Detection System integrating Autoencoder-based anomaly detection with supervised machine learning models.

---

## 📌 Abstract

Financial fraud detection remains a challenging task due to extreme class imbalance and continuously evolving fraudulent behavior. Traditional supervised models rely heavily on labeled historical data and often fail to generalize to unseen fraud patterns.

This project presents a **Hybrid Fraud Detection Framework** that combines **unsupervised anomaly detection (Autoencoder)** with **supervised classifiers (Random Forest and XGBoost)**. The Autoencoder is trained exclusively on legitimate transactions to learn normal behavior, and its **reconstruction error is used as a synthetic feature** to enhance fraud sensitivity.

The framework further incorporates **SMOTE-based balancing**, **threshold optimization**, and **SHAP-based explainability**, resulting in a robust, interpretable, and deployment-ready fraud detection system suitable for real-world financial environments.

---

## 🎯 Objectives

- Detect both known and previously unseen fraud patterns
- Improve recall for minority (fraudulent) transactions
- Reduce false negatives without significantly impacting precision
- Provide explainable, regulator-friendly model outputs
- Design a scalable framework suitable for real-world deployment

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart LR
A["Transaction Data"] --> B["Preprocessing & Normalization"]
B --> C["Autoencoder<br>(Trained on Normal Data)"]
C --> D["Reconstruction Error<br>(Anomaly Score)"]
D --> E["Hybrid Features<br>(Original + RE)"]
E --> F["Supervised Models<br>(RF / XGBoost)"]
F --> G["Fraud / Legitimate Prediction"]
````

---

## ⚙️ Tools & Technologies

| Category           | Tools                                |
| ------------------ | ------------------------------------ |
| Language           | Python 3.10+                         |
| Machine Learning   | scikit-learn, XGBoost, TensorFlow    |
| Imbalance Handling | imbalanced-learn (SMOTE)             |
| Explainability     | SHAP                                 |
| Data Processing    | pandas, numpy                        |
| Visualization      | matplotlib, seaborn                  |
| Model Persistence  | joblib, h5                           |
| Dataset            | Kaggle – Credit Card Fraud Detection |

---

## ✅ Work Completed

### 1. Autoencoder-Based Anomaly Detection

* Trained Autoencoder exclusively on legitimate transactions
* Fine-tuned latent dimension and dropout for improved anomaly sensitivity
* Generated reconstruction error as an anomaly signal

### 2. Hybrid Feature Engineering

* Combined original transaction features with reconstruction error
* Created hybrid feature space for supervised models

### 3. Supervised Learning

* Trained Random Forest and XGBoost on hybrid features
* Evaluated baseline hybrid models

### 4. SMOTE Integration

* Applied SMOTE only on training data
* Improved minority class (fraud) recall

### 5. Threshold Optimization

* Swept thresholds from 0.01 to 0.99
* Selected operating threshold based on:

  * Maximum recall with precision ≥ 0.80
* Final selected model:

  * **XGB_Hybrid_Final**
  * **Threshold = 0.15**

### 6. Explainability (SHAP)

* Global SHAP summary and bar plots
* Local SHAP force plots for fraud predictions
* Verified that reconstruction error is among top influential features

---

## 📊 Final Model Performance (XGB Hybrid – Tuned Threshold)

**Selected Model:** XGB_Hybrid_Final
**Operating Threshold:** 0.15

**Confusion Matrix:**

```
 [56844   20]
 [   14   84]
```

| Metric    | Fraud Class |
| --------- | ----------- |
| Precision | 0.81        |
| Recall    | 0.86        |
| F1-Score  | 0.83        |
| ROC-AUC   | 0.97        |
| PR-AUC    | 0.87        |

✅ Improved recall without significant loss in precision
✅ Suitable for real-world fraud monitoring systems

---

## 📈 Key Visualizations

* ROC Curve and Precision–Recall Curve
* Hybrid vs SMOTE comparison
* Autoencoder reconstruction error distribution
* SHAP feature importance (bar, summary, and force plots)

---

## 🧠 Explainability with SHAP

SHAP analysis confirms that:

* Reconstruction Error is a high-impact feature
* Transaction attributes (V-features and Amount) align with fraud risk
* Model predictions are interpretable and regulator-friendly

---

## 📂 Repository Structure

```
Hybrid-Fraud-Detection-Framework/
│
├── main_notebook/
│   ├── Hybrid_Fraud_Detection.ipynb
│   ├── autoencoder_tuned_model.h5
│   ├── rf_hybrid_final.joblib
│   ├── xgb_hybrid_final.json
│   ├── rf_hybrid_smote.joblib
│   ├── xgb_hybrid_smote.json
│
├── data/
│   ├── X_train_hybrid.csv
│   ├── X_test_hybrid.csv
│   ├── y_train.csv
│   ├── y_test.csv
│
├── results/
│   ├── Hybrid_vs_SMOTE.png
│   ├── ROC_PR_Curves.png
│   ├── SHAP_Summary.png
│
└── README.md
```

---

## 🔮 Future Scope

* Deployment using FastAPI or Streamlit
* Real-time transaction streaming
* Automated retraining and concept drift detection
* Fraud monitoring dashboard

---

## 🧾 Citation

J. Nakar (2025)
**Hybrid Fraud Detection Framework using Autoencoders, SMOTE, and Explainable Machine Learning**

---

## 🙏 Acknowledgement

Developed under the guidance of **Prof. Aswathy Nair**
Department of Computer Engineering
**Marwadi University, India**

```
