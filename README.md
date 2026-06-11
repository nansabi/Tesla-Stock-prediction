📈 Tesla Stock Price Prediction using LSTM & SimpleRNN
🚀 Project Overview

This project is a deep learning-based time series forecasting system that predicts Tesla (TSLA) stock prices using historical stock market data.

It compares two neural network models:

🧠 SimpleRNN (baseline model)
🧠 LSTM (optimized model with hyperparameter tuning)

A Streamlit web app is built to visualize data and generate live predictions.

🎯 Objective

To predict future Tesla stock prices using:

Historical stock data
Deep learning models (RNN & LSTM)
Feature engineering and time-series analysis

📊 Dataset
File: TSLA.csv
Source: Historical Tesla stock price data
Features:
Date
Open
High
Low
Close
Volume

⚙️ Tech Stack
Python 🐍
TensorFlow / Keras 🧠
Pandas & NumPy 📊
Scikit-learn 🔧
Plotly 📈
Streamlit 🌐
Keras Tuner 🔍

🧠 Models Used
1. SimpleRNN (Baseline Model)
Used for initial time-series prediction
Architecture:
SimpleRNN layers
Dense layers
Dropout for regularization

3. LSTM (Optimized Model)
Better at learning long-term dependencies
Tuned using Keras Tuner
Architecture:
LSTM layers (100 units)
Dropout layers
Dense output layer

🔧 Data Preprocessing
Converted Date column to datetime
Sorted data chronologically
Handled missing values using forward/backward fill
Feature engineering:
Daily returns
Moving averages (7, 30 days)
Volatility
Scaled data using MinMaxScaler
Created sequences using 60-day window

📦 Model Training
Input shape: (samples, 60, 1)
Train/Test split: 80% / 20%
Loss function: Mean Squared Error (MSE)
Optimizer: Adam
Callbacks:
EarlyStopping
ModelCheckpoint
ReduceLROnPlateau

📊 Performance Results
Model	RMSE	MAE	MAPE
SimpleRNN	~29.45	~18.02	~5.52%
LSTM (Tuned)	~18.21	~11.41	~3.51%

👉 LSTM performed significantly better than SimpleRNN.

🧪 Hyperparameter Tuning

Used Keras Tuner to optimize:

Number of LSTM units
Dropout rate
Learning rate

Best model saved as:

best_tuned_model.keras
📊 Streamlit Dashboard

Run the app using:

streamlit run app.py

Features:
📌 Real-time stock price visualization
📊 Moving averages (MA7, MA30, MA90)
🔮 Future price prediction (1, 5, 10 days)
📉 Model comparison metrics
📅 Historical data viewer
📁 Project Structure

PROJECT-3-LABMENTIX/
│
├── app.py
├── TSLA.csv
├── model_comparison_results.csv
│
├── best_simplernn_model.keras
├── best_tuned_model.keras
├── best_tuned_model.h5
│
├── lstm_tuning/
├── __pycache__/
└── .venv/


📌 Key Concepts Used
Time Series Forecasting
LSTM (Long Short-Term Memory Networks)
SimpleRNN
Feature Engineering
Data Normalization
Sequence Modeling
Hyperparameter Tuning
📈 Results Summary
LSTM significantly outperformed SimpleRNN
Model achieved ~3.5% prediction error (MAPE)
System successfully predicts short-term Tesla stock trends
⚠️ Limitations
Stock market is highly unpredictable
Model works better for short-term trends
External factors (news, politics) are not included
🚀 Future Improvements
Add sentiment analysis (news + Twitter)
Use Transformer models
Deploy on cloud (AWS / Streamlit Cloud)
Add real-time stock API integration

👨‍💻 Author

Abihail Nans Kuiper Y

AI/ML Project
Deep Learning for Time Series Forecasting
📜 License

