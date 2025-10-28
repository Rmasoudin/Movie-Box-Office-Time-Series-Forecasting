# 🎬 Movie Weekly Box Office Revenue Forecasting

An end-to-end machine learning project that predicts weekly movie box office revenue using time series modeling.  
The project integrates data engineering, MLOps automation, and deep learning to deliver reliable and scalable forecasts for the film industry.

---

## 📊 Overview

This project focuses on **forecasting weekly box office revenue** using real-world movie data scraped from [Box Office Mojo](https://www.boxofficemojo.com/).  
It combines **data collection**, **exploratory analysis**, **feature engineering**, and **model deployment** under a complete MLOps workflow.

---

## 🧱 Project Pipeline

### **Phase 1 – Data Collection & Visualization**
- Scraped movie-level and weekly revenue data from Box Office Mojo.
- Integrated static and dynamic movie features (title, genre, release date, ratings, runtime, revenue, etc.).
- Performed in-depth EDA to identify trends, seasonal patterns, and anomalies.

### **Phase 2 – Data Storage & Preprocessing**
- Designed and stored datasets in **SQLite databases** (train & test).
- Built an **automated preprocessing pipeline** to ensure consistent transformations:
  - Currency and date normalization  
  - Categorical encoding (genres, MPAA ratings)  
  - Feature scaling and normalization  
  - Feature engineering with Fourier series for seasonality  

- Implemented **CI/CD** using **Docker** and **Kubernetes** for reproducibility and scalability.

### **Phase 3 – Modeling & Forecasting**
- Developed a **Deep Neural Network** with multiple dense layers and dropout for regularization.
- Compared performance against a **Naive Baseline** (last week → this week revenue).
- Metrics:
  - **Model:** MSE = 0.1429, MAPE = 2.09%  
  - **Baseline:** MSE = 1.2406, MAPE = 5.40%  
  - ✅ Nearly **10× improvement** in error variance and **2.5× better accuracy**.

---

## ⚙️ Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Data Source** | Box Office Mojo |
| **Data Storage** | SQLite |
| **Data Processing** | Python, Pandas, NumPy, Scikit-learn |
| **Modeling** | TensorFlow / Keras |
| **MLOps & Deployment** | Docker, Kubernetes, MLflow |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git & GitHub |

---

## 🧠 Key Features

- Automated ETL pipeline for train/test datasets  
- Modular data preprocessing and transformation  
- Time series–ready dataset with engineered lag and rolling features  
- Fourier series encoding for seasonal patterns  
- MLflow integration for model tracking  
- Fully containerized and scalable via Docker + Kubernetes  

---

## 📈 Results

| Model | MSE | MAPE |
|--------|------|------|
| **Baseline (Naive Forecast)** | 1.2406 | 5.40% |
| **Deep Neural Network** | **0.1429** | **2.09%** |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Build and run with Docker
docker build -t movie-forecast .
docker run -p 8080:8080 movie-forecast

# Or deploy via Kubernetes
kubectl apply -f deployment.yaml
