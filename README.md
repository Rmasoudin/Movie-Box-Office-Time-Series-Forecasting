# 🎬 Movie Weekly Box Office Revenue Forecasting

An end-to-end data science and MLOps project that predicts weekly movie box office revenue using time series modeling.  
The project combines **data engineering**, **Power BI analytics**, **machine learning**, and **automated deployment** to deliver accurate and scalable forecasting for the film industry.

---

## 📊 Overview

This project aims to forecast **weekly box office revenue** using data scraped from [Box Office Mojo](https://www.boxofficemojo.com/).  
It integrates **data collection**, **exploratory analysis**, **Power BI dashboards**, **feature engineering**, and **model deployment** into one automated pipeline.

---

## 🧱 Project Pipeline

### **Phase 1 – Data Collection & Visualization**
- Scraped **movie-level** and **weekly revenue** data from Box Office Mojo.
- Combined static and dynamic features such as genre, rating, runtime, and weekly earnings.
- Conducted extensive **EDA (Exploratory Data Analysis)** in both **Python** and **Power BI**:
  - Identified revenue trends across genres, months, and release seasons.
  - Visualized performance patterns using **interactive Power BI dashboards**.
  - Discovered key insights about holiday releases, genre popularity, and release timing.

### **Phase 2 – Data Storage & Preprocessing**
- Designed two **SQLite databases** — one for training and one for testing.
- Built a **data preprocessing pipeline** to ensure reproducibility:
  - Currency and date normalization  
  - Categorical encoding (genres, MPAA ratings)  
  - Feature scaling and standardization  
  - Fourier series transformation for seasonality  
  - Log-transformed and rolling window features for trend smoothing
- Automated pipeline orchestration with **Docker** and **Kubernetes** for scalable execution.

### **Phase 3 – Modeling, Forecasting & Model Tracking**
- Developed a **Deep Neural Network** (DNN) to predict weekly box office revenue.
- Compared performance against a **Naive Baseline Model** (previous week → next week revenue).
- Integrated **MLflow** to:
  - Track experiment metrics (MSE, MAPE, loss curves)
  - Log model parameters and artifacts
  - **Store trained models online** for versioning and deployment
- Performance:
  - **Model:** MSE = 0.1429, MAPE = 2.09%  
  - **Baseline:** MSE = 1.2406, MAPE = 5.40%  
  - ✅ Nearly **10× improvement in MSE** and **2.5× better accuracy**.

---

## ⚙️ Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Data Source** | Box Office Mojo |
| **Data Storage** | SQLite |
| **Data Processing** | Python, Pandas, NumPy, Scikit-learn |
| **Visualization & Analytics** | Power BI, Matplotlib, Seaborn |
| **Modeling** | TensorFlow / Keras |
| **MLOps & Deployment** | Docker, Kubernetes, MLflow |
| **Version Control** | Git & GitHub |

---

## 🧠 Key Features

- Automated ETL pipeline for train/test datasets  
- Time series–ready dataset with lag, rolling, and Fourier features  
- Power BI dashboard for interactive revenue insights  
- Deep learning model for accurate weekly revenue prediction  
- MLflow integration for experiment tracking  
- Fully containerized and deployable with Docker + Kubernetes  

---

## 📈 Results

| Model | MSE | MAPE |
|--------|------|------|
| **Baseline (Naive Forecast)** | 1.2406 | 5.40% |
| **Deep Neural Network** | **0.1429** | **2.09%** |

---

## 🧩 Power BI Insights

Power BI was used for:
- Exploring **genre-wise and studio-wise revenue distributions**
- Tracking **weekly and cumulative earnings over time**
- Highlighting **holiday release performance** and **seasonal spikes**
- Visualizing **revenue change trends** and **prediction comparisons**

Power BI dashboards helped validate EDA findings and present business-friendly visual insights.

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

