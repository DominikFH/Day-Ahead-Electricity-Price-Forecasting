
# Day-Ahead Electricity Price Forecasting

This project aims to forecast **day-ahead electricity prices** using historical market data, weather data, and machine learning models. It focuses on building robust time series forecasting pipelines that can help anticipate electricity price fluctuations in advance.

---

## 🔍 Project Overview

- **Goal**: Predict day-ahead electricity prices (e.g., EPEX Spot) for improved planning and cost optimization.
- **Data Sources**:
  - Historical electricity and other comodity prices
  - Demand and supply indicators

- **Methods & Models**:
  - Deep learning: LSTM, CNN, TFT

---

## 📂 Repository Structure

```
.
├── data/               # Raw and processed datasets (excluded from git)
├── jupyter_book/       # Documentation in german language
├── notebooks/          # Jupyter notebooks for training and forecasting
├── src/                # Python scripts for data prep
├── requirements.txt    # Python dependencies
├── README.md           # Project description
└── .gitignore          # Ignore patterns for git
```

---

## 🚀 How to Run

1️⃣ Clone the repository:
```bash
git clone https://github.com/DominikFH/Day-Ahead-Electricity-Price-Forecasting.git
cd Day-Ahead-Electricity-Price-Forecasting
```

2️⃣ Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

4️⃣ Explore the Jupyter notebooks:
```bash
jupyter notebook
```

5️⃣ Explore the documentation:
```bash
jupyter notebook
---

## 📊 Results

- Evaluation metrics: RMSE, MAE, MAPE
- Baseline comparison: naive models vs. advanced models
- Example forecast plots (see notebooks)

---

## 📅 Future Work

- Incorporate real-time weather forecasts
- Add LSTM or transformer-based deep learning models
- Optimize hyperparameters with grid search
- Deploy as an API for live predictions

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

- **Author**: DominikFH  
- GitHub: [DominikFH](https://github.com/DominikFH)
