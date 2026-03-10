Financial Time Series Prediction with CNN-LSTM

This project explores the use of deep learning for predicting short-term movements in financial markets. Using historical price data from the QQQ ETF, the model attempts to classify the next day's movement as Up, Down, or Neutral based on a set of technical indicators derived from historical prices.

The goal of the project was not to build a profitable trading system, but to explore the challenges of applying machine learning to financial time series data.

Project Overview

The project follows a typical machine learning workflow for time series data.

1. Data Collection

Historical daily market data is downloaded using the yfinance library for the QQQ ETF, which tracks the Nasdaq-100 index.

2. Feature Engineering

Several technical indicators are generated using pandas_ta. These include:

Simple and exponential moving averages

Bollinger Bands

RSI

MACD

Stochastic oscillator

ATR

rolling volatility

log returns and derived statistics

Daily returns are calculated and categorized into three classes using a small threshold to filter minor price fluctuations.

3. Label Creation

Each day is classified into one of three categories:

Up – return greater than a defined positive threshold

Down – return lower than a defined negative threshold

Neutral – small movements within the threshold range

These labels are converted into one-hot encoded targets for the model.

4. Normalization

The dataset features are standardized using the mean and standard deviation of the dataset. This helps stabilize neural network training and prevents features with larger scales from dominating the model.

5. Sequence Creation

Since financial data is sequential, the dataset is converted into rolling time windows.
Each training sample consists of 60 consecutive days of features, and the model predicts the movement of the following day.

6. Model Architecture

The model combines convolutional and recurrent neural network layers:

Conv1D layer to capture short-term patterns in the time series

Bidirectional LSTM layers to model temporal dependencies

Dense layers for final classification

Class imbalance between the movement categories is handled using class weights during training.

7. Evaluation

Model performance is evaluated using:

classification report

confusion matrix

Results and Observations

The model did not achieve meaningful predictive performance. The predictions were close to random and the model struggled to consistently distinguish between the three movement classes.

This result highlights a common issue in financial machine learning: technical indicators derived purely from historical prices often do not contain enough predictive information for reliable short-term forecasting.

In practice, stronger models often rely on additional external data sources, such as:

macroeconomic indicators

interest rates

news sentiment

earnings reports

options market data

market microstructure signals

This project therefore serves as an exploration of the pipeline and challenges involved in financial time series modeling rather than a successful prediction system.

Technologies Used

Python
TensorFlow / Keras
Pandas
NumPy
pandas_ta
yfinance
Scikit-learn
Matplotlib
Seaborn

Repository Structure

data_generation.py
Downloads historical market data and generates technical indicators and movement labels.

dataset_normalization.py
Normalizes the dataset using mean and standard deviation.

model_training.py
Builds the CNN-LSTM model, trains it on sequential data, and evaluates predictions.

environment_check.py
Utility script for verifying installed TensorFlow and related library versions.

Purpose of the Project

The purpose of this project was to experiment with deep learning techniques applied to financial time series data and to build a complete machine learning pipeline from raw market data to model evaluation.

It also illustrates the practical difficulties of predicting financial markets using historical price-based indicators alone.
