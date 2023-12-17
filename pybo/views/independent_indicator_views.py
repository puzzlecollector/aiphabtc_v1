# Standard library imports
import sys
import os
import csv
import time
import json
import asyncio
from io import StringIO
from datetime import datetime, timedelta, timezone
import pickle
import openai

# Third-party libraries imports
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import pyupbit
import faiss
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
import pandas_ta as ta
from transformers import AutoModel, AutoTokenizer, AutoConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification
from tokenization_roberta_spm import FairSeqRobertaSentencePieceTokenizer
import plotly.graph_objects as go
from plotly.io import to_html
import plotly.io as pio
from plotly.graph_objs import Scatter
import plotly.express as px
from bs4 import BeautifulSoup
from dateutil import parser
from tqdm import tqdm
import zoneinfo
import joblib

# Django specific imports
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.cache import cache

def preprocess(df):
    # Convert timestamps to datetime objects in Korean timezone
    korean_timezone = zoneinfo.ZoneInfo("Asia/Seoul")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert(korean_timezone)
    return df

def basic_aiphabot_1hr():
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", "1h")
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    hours, days, months = [], [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        hour = dtobj.hour
        day = dtobj.day
        month = dtobj.month
        hours.append(hour)
        days.append(day)
        months.append(month)
    chart_df["hours"] = hours
    chart_df["days"] = days
    chart_df["months"] = months
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)

    chart_df["bop"] = chart_df.ta.bop(lookahead=False)
    chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False)
    chart_df["cmf"] = chart_df.ta.cmf(lookahead=False)
    chart_df["vwap"] = chart_df.ta.vwap(lookahead=False)
    chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100.0
    chart_df["high/low"] = chart_df["high"] / chart_df["low"]
    chart_df["low/open"] = chart_df["low"] / chart_df["open"]
    chart_df["hwma"] = chart_df.ta.hwma(lookahead=False)
    chart_df["hwma/close"] = chart_df["hwma"] / chart_df["close"]

    for l in range(1, 12):
        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            val = chart_df[col].values
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i - l] == 0:
                    ret = 1
                else:
                    ret = val[i] / val[i - l]
                val_ret.append(ret)
            chart_df["{}_change_{}".format(col, l)] = val_ret

    enter_price = chart_df["open"].values[-1]
    long_take_profit_price = enter_price * (1 + 0.50/100)
    short_take_profit_price = enter_price * (1 - 0.50/100)

    chart_df.drop(columns={"timestamp", "open", "high", "low", "close", "volume", "vwap", "hwma", "datetime"}, inplace=True)
    chart_df.dropna(inplace=True)

    date_obj = chart_df.index[-1]
    date_obj_end = date_obj + pd.Timedelta(hours=1)
    # add 9 hours fix
    # date_obj = date_obj + pd.Timedelta(hours=9)
    # date_obj_end = date_obj_end + pd.Timedelta(hours=9)

    date_obj = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    date_obj_end = date_obj_end.strftime('%Y-%m-%d %H:%M:%S')
    # make prediction
    model = XGBClassifier()
    model.load_model("pybo/views/aiphabot_1hr")
    x = chart_df.iloc[-2, :].values.reshape((-1, chart_df.shape[1]))  # get previous timestep information
    pred = model.predict_proba(x)
    pred = pred[0]
    return date_obj, date_obj_end, pred, enter_price, long_take_profit_price, short_take_profit_price

def basic_aiphabot_4hrs():
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", "4h")
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    hours, days, months = [], [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        hour = dtobj.hour
        day = dtobj.day
        month = dtobj.month
        hours.append(hour)
        days.append(day)
        months.append(month)
    chart_df["hours"] = hours
    chart_df["days"] = days
    chart_df["months"] = months
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)

    chart_df["bop"] = chart_df.ta.bop(lookahead=False)
    chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False)
    chart_df["cmf"] = chart_df.ta.cmf(lookahead=False)
    chart_df["vwap"] = chart_df.ta.vwap(lookahead=False)
    chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100.0
    chart_df["high/low"] = chart_df["high"] / chart_df["low"]
    chart_df["low/open"] = chart_df["low"] / chart_df["open"]
    chart_df["hwma"] = chart_df.ta.hwma(lookahead=False)
    chart_df["hwma/close"] = chart_df["hwma"] / chart_df["close"]
    for l in range(1, 12):
        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            val = chart_df[col].values
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i - l] == 0:
                    ret = 1
                else:
                    ret = val[i] / val[i - l]
                val_ret.append(ret)
            chart_df["{}_change_{}".format(col, l)] = val_ret

    enter_price = chart_df["open"].values[-1]
    long_take_profit_price = enter_price * (1 + 0.75 / 100)
    short_take_profit_price = enter_price * (1 - 0.75 / 100)

    chart_df.drop(columns={"timestamp", "open", "high", "low", "close", "volume", "vwap", "hwma", "datetime"}, inplace=True)
    chart_df.dropna(inplace=True)
    date_obj = chart_df.index[-1]
    date_obj_end = date_obj + pd.Timedelta(hours=4)
    # add 9 hours fix
    # date_obj = date_obj + pd.Timedelta(hours=9)
    # date_obj_end = date_obj_end + pd.Timedelta(hours=9)

    date_obj = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    date_obj_end = date_obj_end.strftime('%Y-%m-%d %H:%M:%S')
    # make prediction
    model = XGBClassifier()
    model.load_model("pybo/views/xgb_optuna_optimized_4hrs")
    x = chart_df.iloc[-2, :].values.reshape((-1, chart_df.shape[1]))  # get previous timestep information
    pred = model.predict_proba(x)
    pred = pred[0]
    return date_obj, date_obj_end, pred, enter_price, long_take_profit_price, short_take_profit_price

def basic_aiphabot_1d():
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", "1d")
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    hours, days, months = [], [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        hour = dtobj.hour
        day = dtobj.day
        month = dtobj.month
        hours.append(hour)
        days.append(day)
        months.append(month)
    chart_df["hours"] = hours
    chart_df["days"] = days
    chart_df["months"] = months
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)

    chart_df["bop"] = chart_df.ta.bop(lookahead=False)
    chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False)
    chart_df["cmf"] = chart_df.ta.cmf(lookahead=False)
    chart_df["vwap"] = chart_df.ta.vwap(lookahead=False)
    chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100.0
    chart_df["high/low"] = chart_df["high"] / chart_df["low"]
    chart_df["low/open"] = chart_df["low"] / chart_df["open"]
    chart_df["hwma"] = chart_df.ta.hwma(lookahead=False)
    chart_df["hwma/close"] = chart_df["hwma"] / chart_df["close"]
    for l in range(1, 12):
        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            val = chart_df[col].values
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i - l] == 0:
                    ret = 1
                else:
                    ret = val[i] / val[i - l]
                val_ret.append(ret)
            chart_df["{}_change_{}".format(col, l)] = val_ret

    enter_price = chart_df["open"].values[-1]
    long_take_profit_price = enter_price * (1 + 3 / 100)
    short_take_profit_price = enter_price * (1 - 3 / 100)

    chart_df.drop(columns={"timestamp", "open", "high", "low", "close", "volume", "vwap", "hwma", "datetime"}, inplace=True)
    chart_df.dropna(inplace=True)
    date_obj = chart_df.index[-1]
    date_obj_end = date_obj + pd.Timedelta(hours=24)
    # add 9 hours fix
    # date_obj = date_obj + pd.Timedelta(hours=9)
    # date_obj_end = date_obj_end + pd.Timedelta(hours=9)

    date_obj = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    date_obj_end = date_obj_end.strftime('%Y-%m-%d %H:%M:%S')
    # make prediction
    model = XGBClassifier()
    model.load_model("pybo/views/aiphabot_1d")
    x = chart_df.iloc[-2, :].values.reshape((-1, chart_df.shape[1]))  # get previous timestep information
    pred = model.predict_proba(x)
    pred = pred[0]
    return date_obj, date_obj_end, pred, enter_price, long_take_profit_price, short_take_profit_price

def aiphabot_15mins():
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", "15m")
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    minutes, hours, days, months = [], [], [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        minute = dtobj.minute
        hour = dtobj.hour
        day = dtobj.day
        month = dtobj.month
        minutes.append(minute)
        hours.append(hour)
        days.append(day)
        months.append(month)
    chart_df["minutes"] = minutes
    chart_df["hours"] = hours
    chart_df["days"] = days
    chart_df["months"] = months
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)
    chart_df["bop"] = chart_df.ta.bop(lookahead=False)
    chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False)
    chart_df["cmf"] = chart_df.ta.cmf(lookahead=False)
    chart_df["vwap"] = chart_df.ta.vwap(lookahead=False)
    chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100.0
    chart_df["high/low"] = chart_df["high"] / chart_df["low"]
    chart_df["low/open"] = chart_df["low"] / chart_df["open"]
    chart_df["hwma"] = chart_df.ta.hwma(lookahead=False)
    chart_df["hwma/close"] = chart_df["hwma"] / chart_df["close"]
    for l in range(1, 12):
        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            val = chart_df[col].values
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i - l] == 0:
                    ret = 1
                else:
                    ret = val[i] / val[i - l]
                val_ret.append(ret)
            chart_df["{}_change_{}".format(col, l)] = val_ret

    enter_price = chart_df["open"].values[-1]

    chart_df.drop(columns={"timestamp", "open", "high", "low", "close", "volume", "vwap", "hwma", "datetime"}, inplace=True)
    chart_df.dropna(inplace=True)

    date_obj = chart_df.index[-1]
    date_obj_end = date_obj + pd.Timedelta(minutes=15)
    # add 9 hours fix # unnecessary
    # date_obj = date_obj + pd.Timedelta(hours=9)
    # date_obj_end = date_obj_end + pd.Timedelta(hours=9)

    date_obj = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    date_obj_end = date_obj_end.strftime('%Y-%m-%d %H:%M:%S')
    # make prediction
    model = XGBClassifier()
    model.load_model("pybo/views/15min_xgb_v1")
    x = chart_df.iloc[-2, :].values.reshape((-1, chart_df.shape[1]))  # get previous timestep information
    pred = model.predict_proba(x)
    pred = pred[0]
    return date_obj, date_obj_end, pred, enter_price

def get_kimchi_data():
    bitget = ccxt.bitget()
    data = {}
    data["current_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    USDKRW = yf.Ticker("USDKRW=X")
    history = USDKRW.history(period="1d")
    data["now_usd_krw"] = history["Close"].iloc[0]
    data["now_upbit_price"] = pyupbit.get_current_price("KRW-BTC")
    data["now_bitget_price"] = bitget.fetch_ticker("BTC/USDT")["close"]
    data["kp"] = round((data["now_upbit_price"] * 100 / (data["now_bitget_price"] * data["now_usd_krw"])) - 100, 3)
    return data

def get_correlations():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    # Fetch data for S&P 500, Bitcoin, Gold, and Ethereum
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
    bitcoin = yf.download('BTC-USD', start=start_date, end=end_date)['Close']
    gold = yf.download('GC=F', start=start_date, end=end_date)['Close']
    ethereum = yf.download('ETH-USD', start=start_date, end=end_date)['Close']
    # Align the lengths of the data
    min_length = min(len(sp500), len(bitcoin), len(gold), len(ethereum))
    sp500_aligned = sp500.iloc[:min_length]
    bitcoin_aligned = bitcoin.iloc[:min_length]
    gold_aligned = gold.iloc[:min_length]
    ethereum_aligned = ethereum.iloc[:min_length]
    # Create a DataFrame with aligned data
    data_aligned = pd.DataFrame(
        {'SP500': sp500_aligned, 'Bitcoin': bitcoin_aligned, 'Gold': gold_aligned, 'Ethereum': ethereum_aligned})
    # Calculate Pearson, Spearman, and Kendall Tau correlations
    pearson_corr = data_aligned.corr(method='pearson')
    spearman_corr = data_aligned.corr(method='spearman')
    kendall_corr = data_aligned.corr(method='kendall')
    pearson_corr_dict = pearson_corr.to_dict(orient='index')
    spearman_corr_dict = spearman_corr.to_dict(orient='index')
    kendall_corr_dict = kendall_corr.to_dict(orient='index')
    return pearson_corr_dict, spearman_corr_dict, kendall_corr_dict

def get_predictions_fbprophet(btc_sequence, timestamps, timeframe):
    df = pd.DataFrame({
        'ds': timestamps,
        'y': btc_sequence,
    })
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=6, freq='1H')
    if timeframe == "1h":
        future = model.make_future_dataframe(periods=6, freq="1H")
    elif timeframe == "4h":
        future = model.make_future_dataframe(periods=6, freq="4H")
    elif timeframe == "1d":
        future = model.make_future_dataframe(periods=6, freq="1D")
    # future = model.make_future_dataframe(periods=6, freq='30min')
    forecast = model.predict(future)
    yhat = forecast['yhat'].iloc[-6:].values
    return yhat

def get_predictions_arima(btc_sequence, p=1, d=1, q=1, steps_ahead=6):
    try:
        # Differencing
        btc_diff = np.diff(btc_sequence, n=d)
        # Fit ARIMA model
        model = ARIMA(btc_diff, order=(p, 0, q))
        fitted_model = model.fit()
        # Forecast
        forecast_diff = fitted_model.forecast(steps=steps_ahead)
        # Invert differencing
        forecast = [btc_sequence[-1]]
        for diff in forecast_diff:
            forecast.append(forecast[-1] + diff)
        return forecast[1:]
    except Exception as e:
        print(f"Model fitting failed: {str(e)}")
        return np.zeros((steps_ahead,))

def time_series_analysis(request, timeframe='1h'):
    bitget = ccxt.bitget()
    btcusdt = bitget.fetch_ohlcv("BTC/USDT:USDT", timeframe)
    btc = pd.DataFrame(btcusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_close = btc['close'].values
    labels = [
        datetime.utcfromtimestamp(t / 1000)
        .replace(tzinfo=timezone.utc)
        .astimezone(zoneinfo.ZoneInfo('Asia/Seoul'))  # Use ZoneInfo instead of timezone
        .strftime('%Y-%m-%d %H:%M:%S')
        for t in btc['timestamp']
    ]
    # Get predictions
    prophet_forecast = get_predictions_fbprophet(btc_close, labels, timeframe)
    arima_forecast = get_predictions_arima(btc_close)
    # Convert numpy float32 to Python float for JSON serialization
    prophet_forecast = [float(f) for f in prophet_forecast]
    arima_forecast = [float(f) for f in arima_forecast]
    # Append labels for forecast
    last_label = labels[-1]
    last_timestamp = datetime.strptime(last_label, '%Y-%m-%d %H:%M:%S')
    if timeframe == '1h':
        interval = timedelta(minutes=60)
    elif timeframe == '4h':
        interval = timedelta(minutes=240)
    elif timeframe == '1d':
        interval = timedelta(minutes=1440)
    # Generate next 6 timestamps
    new_labels = [last_timestamp + interval * i for i in range(1, 7)]
    new_labels_str = [t.strftime('%Y-%m-%d %H:%M:%S') for t in new_labels]
    # Append new labels to existing labels
    labels.extend(new_labels_str)
    context = {
        "input_seq": [float(f) for f in list(btc_close[-30:])],  # Convert numpy float32 to Python float
        "prophet_forecast": prophet_forecast,
        "arima_forecast": arima_forecast,
        "labels": labels[-36:],
    }
    return JsonResponse(context)

def get_correlations():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    # Fetch data for S&P 500, Bitcoin, Gold, and Ethereum
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
    bitcoin = yf.download('BTC-USD', start=start_date, end=end_date)['Close']
    gold = yf.download('GC=F', start=start_date, end=end_date)['Close']
    ethereum = yf.download('ETH-USD', start=start_date, end=end_date)['Close']
    # Align the lengths of the data
    min_length = min(len(sp500), len(bitcoin), len(gold), len(ethereum))
    sp500_aligned = sp500.iloc[:min_length]
    bitcoin_aligned = bitcoin.iloc[:min_length]
    gold_aligned = gold.iloc[:min_length]
    ethereum_aligned = ethereum.iloc[:min_length]
    # Create a DataFrame with aligned data
    data_aligned = pd.DataFrame(
        {'SP500': sp500_aligned, 'Bitcoin': bitcoin_aligned, 'Gold': gold_aligned, 'Ethereum': ethereum_aligned})
    # Calculate Pearson, Spearman, and Kendall Tau correlations
    pearson_corr = data_aligned.corr(method='pearson')
    spearman_corr = data_aligned.corr(method='spearman')
    kendall_corr = data_aligned.corr(method='kendall')
    pearson_corr_dict = pearson_corr.to_dict(orient='index')
    spearman_corr_dict = spearman_corr.to_dict(orient='index')
    kendall_corr_dict = kendall_corr.to_dict(orient='index')
    return pearson_corr_dict, spearman_corr_dict, kendall_corr_dict

def get_technical_indicators(timeframe='1h'):
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['STD_20'] = df['close'].rolling(window=20).std()
    df['Upper_Bollinger'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_Bollinger'] = df['SMA_20'] - (df['STD_20'] * 2)
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    # get last four rows
    sample = df.iloc[-4:, 1:]
    sample_str = sample.to_string(index=False)
    data = {"output_str":sample_str}
    return data

def fetch_ai_technical1h(request):
    technical_data = get_technical_indicators(timeframe='1h')
    technical_output = technical_data["output_str"]
    # Craft the message
    message = ("다음과 같은 1시간봉 비트코인 데이터가 주어졌을때:\n\n"
               "{}\n\n"
               "비트코인 가격 추세를 분석하고 총평을 해줘."
               ).format(technical_output)
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.5
    )
    chat_message = response["choices"][0]["message"]["content"]
    return JsonResponse({'chat_message': chat_message})

def fetch_ai_technical4h(request):
    technical_data = get_technical_indicators(timeframe='4h')
    technical_output = technical_data["output_str"]
    # Craft the message
    message = ("다음과 같은 4시간봉 비트코인 데이터가 주어졌을때:\n\n"
               "{}\n\n"
               "비트코인 가격 추세를 분석하고 총평을 해줘."
               ).format(technical_output)
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.5
    )
    chat_message = response["choices"][0]["message"]["content"]
    return JsonResponse({'chat_message': chat_message})

def fetch_ai_technical1d(request):
    technical_data = get_technical_indicators(timeframe='1d')
    technical_output = technical_data["output_str"]
    # Craft the message
    message = ("다음과 같은 일봉 비트코인 데이터가 주어졌을때:\n\n"
               "{}\n\n"
               "비트코인 가격 추세를 분석하고 총평을 해줘."
               ).format(technical_output)
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.5
    )
    chat_message = response["choices"][0]["message"]["content"]
    return JsonResponse({'chat_message': chat_message})

def fetch_ai_analysis(request, timeframe):
    if timeframe == '1h':
        return fetch_ai_technical1h(request)
    elif timeframe == '4h':
        return fetch_ai_technical4h(request)
    elif timeframe == '1d':
        return fetch_ai_technical1d(request)
    else:
        return JsonResponse({'chat_message': 'Invalid timeframe.'})

# for coinness data scraping
def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "html.parser")
    title = soup.find("span", {"class": "view_top_title noselect"}).text.strip()
    content = soup.find("div", {"class": "view_text noselect"}).get_text(strip=True)
    content = content.replace("이 광고는 쿠팡 파트너스 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받습니다.", "")
    return title, content
def scrape_tokenpost():
    all_titles, all_contents, all_full_times = [], [], []
    for i in tqdm(range(1, 2), desc="Scraping content from tokenpost"):
        try:
            links = []
            url = f"https://www.tokenpost.kr/coinness?page={i}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            news_req = requests.get(url, headers=headers)
            soup = BeautifulSoup(news_req.content, "html.parser")
            elems = soup.find_all("div", {"class": "list_left_item"})
            for e in elems:
                article_elems = e.find_all("div", {"class": "list_item_text"})
                for article in article_elems:
                    title_link = article.find("a", href=True)
                    if title_link and '/article-' in title_link['href']:
                        full_link = 'https://www.tokenpost.kr' + title_link['href']
                        # Find the date element in the parent of the article
                        date_elem = article.parent.find("span", {"class": "day"})
                        news_date = parser.parse(date_elem.text)
                        links.append(full_link)
                        all_full_times.append(news_date)
                    if len(all_full_times) > 4:
                        break
            for link in links:
                try:
                    title, content = get_articles(headers, link)
                    all_titles.append(title)
                    all_contents.append(content)
                except Exception as e:
                    print(f"Error while scraping news content: {e}")
        except Exception as e:
            print(f"Error while scraping page {i}: {e}")
        time.sleep(0.1)
    return pd.DataFrame({'titles': all_titles, 'contents': all_contents, 'datetimes': all_full_times})

def get_sentiment_scores(df):
    titles = df["titles"].values
    contents = df["contents"].values
    tokenizer = FairSeqRobertaSentencePieceTokenizer.from_pretrained("pybo/views/fairseq-roberta-all-model")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("pybo/views/cryptoroberta")
    scores = np.zeros(3)
    for i in range(len(titles)):
        encoded_inputs = tokenizer(str(titles[i]), str(contents[i]), max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            sentiment = sentiment_model(**encoded_inputs)[0]
            sentiment = nn.Softmax(dim=1)(sentiment)[0]
            sentiment = sentiment.detach().cpu().numpy()
        scores += sentiment
    scores /= int(df.shape[0])
    return scores # average scores

# get technical voters
def get_technical_voters(timeframe="1h"):
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", timeframe)
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    hours, days, months = [], [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        hour = dtobj.hour
        day = dtobj.day
        month = dtobj.month
        hours.append(hour)
        days.append(day)
        months.append(month)
    chart_df["hours"] = hours
    chart_df["days"] = days
    chart_df["months"] = months
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)

    # RSI
    rsi = chart_df.ta.rsi(length=14)
    rsi_df = rsi.to_frame()
    rsi_vals = rsi_df["RSI_14"].values

    # STOCH
    stoch = chart_df.ta.stoch(high='high', low='low', close='close')
    stoch_vals = stoch["STOCHk_14_3_3"].values

    # STOCHRSI
    stochrsi = chart_df.ta.stochrsi(close='close')
    stochrsi_vals = stochrsi["STOCHRSIk_14_14_3_3"].values

    # MACD
    macd = chart_df.ta.macd(close='close')
    macd_vals = macd["MACD_12_26_9"].values

    # ADX
    adx = chart_df.ta.adx(high='high', low='low', close='close')
    adx_vals = adx['ADX_14'].values

    # Williams %R
    williams_r = chart_df.ta.willr(high='high', low='low', close='close')
    williams_r = williams_r.to_frame()
    williams_r_vals = williams_r["WILLR_14"].values

    # CCI
    cci = chart_df.ta.cci(high='high', low='low', close='close')
    cci = cci.to_frame()
    cci_vals = cci["CCI_14_0.015"].values

    # ATR
    atr = chart_df.ta.atr(high='high', low='low', close='close')
    atr = atr.to_frame()
    atr_vals = atr["ATRr_14"].values

    # Highs/Lows
    highs = chart_df["close"].rolling(window=14).max()
    lows = chart_df["close"].rolling(window=14).min()
    highs_lows_diff = highs - lows
    highs_lows_diff_df = highs_lows_diff.to_frame()
    highs_lows_vals = highs_lows_diff_df["close"].values

    # Ultimate Oscillator
    uo = chart_df.ta.uo(high='high', low='low', close='close')
    uo = uo.to_frame()
    uo_vals = uo["UO_7_14_28"].values

    # ROC
    roc = chart_df.ta.roc(close='close', length=10)
    roc = roc.to_frame()
    roc_vals = roc["ROC_10"].values

    # Bull/Bear Power
    power = chart_df['close'] - chart_df.ta.ema(close='close', length=13)
    power = power.to_frame()
    power_vals = power[0].values

    indicators = {
        'RSI': rsi_vals,
        'STOCH': stoch_vals,
        'STOCHRSI': stochrsi_vals,
        'MACD': macd_vals,
        'ADX': adx_vals,
        'Williams %R': williams_r_vals,
        'CCI': cci_vals,
        'ATR': atr_vals,
        'Highs/Lows': highs_lows_vals,
        'Ultimate Oscillator': uo_vals,
        'ROC': roc_vals,
        'Bull/Bear Power': power_vals
    }

    actions = {
        'RSI': 'Long' if indicators['RSI'][-1] < 30 else 'Short' if indicators['RSI'][-1] > 70 else 'Neutral',
        'STOCH': 'Long' if indicators['STOCH'][-1] < 20 else 'Short' if indicators['STOCH'][-1] > 80 else 'Neutral',
        'STOCHRSI': 'Long' if indicators['STOCHRSI'][-1] < 0.2 else 'Short' if indicators['STOCHRSI'][-1] > 0.8 else 'Neutral',
        'MACD': 'Long' if indicators['MACD'][-1] > 0 else 'Short' if indicators['MACD'][-1] < 0 else 'Neutral',
        'ADX': 'Long' if indicators['ADX'][-1] > 25 else 'Neutral',
        'Williams %R': 'Long' if indicators['Williams %R'][-1] < -80 else 'Short' if indicators['Williams %R'][-1] > -20 else 'Neutral',
        'CCI': 'Long' if indicators['CCI'][-1] < -100 else 'Short' if indicators['CCI'][-1] > 100 else 'Neutral',
        'ATR': 'High Volatility' if indicators['ATR'][-1] > indicators['ATR'][-2] else 'Low Volatility',
        'Highs/Lows': 'Long' if indicators['Highs/Lows'][-1] > 0 else 'Short' if indicators['Highs/Lows'][-1] < 0 else 'Neutral',
        'Ultimate Oscillator': 'Long' if indicators['Ultimate Oscillator'][-1] < 30 else 'Short' if indicators['Ultimate Oscillator'][-1] > 70 else 'Neutral',
        'ROC': 'Long' if indicators['ROC'][-1] > 0 else 'Short' if indicators['ROC'][-1] < 0 else 'Neutral',
        'Bull/Bear Power': 'Long' if indicators['Bull/Bear Power'][-1] > 0 else 'Short' if indicators['Bull/Bear Power'][-1] < 0 else 'Neutral'
    }

    buy_count = sum(1 for action in actions.values() if action == 'Long')
    sell_count = sum(1 for action in actions.values() if action == 'Short')
    neutral_count = sum(1 for action in actions.values() if action == 'Neutral')

    if buy_count > sell_count and buy_count > neutral_count:
        recommended_action = 'Long'
    elif sell_count > buy_count and sell_count > neutral_count:
        recommended_action = 'Short'
    else:
        recommended_action = 'Neutral'

    return actions, recommended_action


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions

# LSTM inference function
def infer_LSTM():
    input_size = 7
    hidden_layer_size = 64
    output_size = 2
    LSTM_test = LSTMClassifier(input_size, hidden_layer_size, output_size)
    checkpoint = torch.load("pybo/views/best_lstm.pt", map_location=torch.device('cpu')) # load model on CPU - may affect performance
    LSTM_test.load_state_dict(checkpoint)
    test_scaler = joblib.load("pybo/views/LSTM_minmax_scaler.pkl")

    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", "1d")
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    days, months = [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        day = dtobj.day
        month = dtobj.month
        days.append(day)
        months.append(month)
    chart_df["month"] = months
    chart_df["day"] = days
    chart_df.drop(columns={"timestamp", "datetime"}, inplace=True)

    test_scaled_features = test_scaler.transform(chart_df)
    test_input = test_scaled_features[-15:-1]  # LSTM uses 14 days
    test_input = test_input.reshape((-1, 14, 7))
    test_input = torch.tensor(test_input, dtype=torch.float32)
    LSTM_test.eval()
    with torch.no_grad():
        output = LSTM_test(test_input)
        probs = nn.Softmax(dim=-1)(output)[0]
        probs = probs.detach().cpu().numpy()
    LSTM_long_prob = round(probs[1] * 100, 2)
    LSTM_short_prob = round(probs[0] * 100, 2)
    return LSTM_long_prob, LSTM_short_prob


# Transformer Encoder Classifier
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, dim_feedforward, output_size):
        super(TransformerClassifier, self).__init__()

        # Embedding layer for positional encoding
        self.embedding = nn.Linear(input_size, input_size)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # Linear layer for final output
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Positional embedding
        x = self.embedding(x)
        # Passing through the transformer layers
        x = self.transformer_encoder(x)
        # Taking the output of the last time step
        x = x[:, -1, :]
        # Output layer
        output = self.fc(x)
        return output

# Transformer Encoder inference function
def infer_Transformer():
    input_size = 7  # As per your dataset
    num_layers = 3  # Number of Transformer layers
    num_heads = 1  # Number of heads in Multi-Head Attention
    dim_feedforward = 512  # Feedforward dimension
    output_size = 2  # Number of output classes

    Transformer_test = TransformerClassifier(input_size, num_layers, num_heads, dim_feedforward, output_size)
    checkpoint = torch.load("pybo/views/best_transformer_encoder.pt", map_location=torch.device('cpu'))
    Transformer_test.load_state_dict(checkpoint)

    test_scaler = joblib.load("pybo/views/Transformer_minmax_scaler.pkl")

    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT:USDT", "1d")
    chart_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chart_df = preprocess(chart_df)
    days, months = [], []
    for dt in chart_df["datetime"]:
        dtobj = pd.to_datetime(dt)
        day = dtobj.day
        month = dtobj.month
        days.append(day)
        months.append(month)
    chart_df["month"] = months
    chart_df["day"] = days
    chart_df.drop(columns={"timestamp", "datetime"}, inplace=True)

    test_scaled_features = test_scaler.transform(chart_df)
    test_input = test_scaled_features[-15:-1]
    test_input = test_input.reshape((-1, 14, 7))
    test_input = torch.tensor(test_input, dtype=torch.float32)

    Transformer_test.eval()
    with torch.no_grad():
        output = Transformer_test(test_input)
        probs = nn.Softmax(dim=-1)(output)[0]
        probs = probs.detach().cpu().numpy()

    Transformer_long_prob = round(probs[1] * 100, 2)
    Transformer_short_prob = round(probs[0] * 100, 2)
    return Transformer_long_prob, Transformer_short_prob


def independent_indicator_view(request):
    aipha_date_obj15m, aipha_date_obj_end15m, aipha_predictions15m, aipha_enter_price15m = aiphabot_15mins()
    aipha_date_obj1h, aipha_date_obj_end1h, aipha_predictions1h, aipha_enter_price1h, aipha_long_take_profit1h, aipha_short_take_profit1h = basic_aiphabot_1hr()
    aipha_date_obj4h, aipha_date_obj_end4h, aipha_predictions4h, aipha_enter_price4h, aipha_long_take_profit4h, aipha_short_take_profit4h = basic_aiphabot_4hrs()
    aipha_date_obj1d, aipha_date_obj_end1d, aipha_predictions1d, aipha_enter_price1d, aipha_long_take_profit1d, aipha_short_take_profit1d = basic_aiphabot_1d()

    url_fng = 'https://api.alternative.me/fng/?limit=7&date_format=kr'
    response_fng = requests.get(url_fng)
    data_fng = response_fng.json().get('data', [])

    url_global = 'https://api.coinlore.net/api/global/'
    response_global = requests.get(url_global)
    data_global = response_global.json()

    kimchi_data = get_kimchi_data()

    pearson_corr, spearman_corr, kendall_corr = get_correlations()

    df = scrape_tokenpost()  # call your scraping function
    avg_sentiment_scores = get_sentiment_scores(df)
    scraped_data = df.to_dict(orient='records')  # convert DataFrame to list of dicts
    avg_sentiment_scores_percentage = [round(score * 100, 2) for score in avg_sentiment_scores]
    sentiment_labels = ['호재', '악재', '중립']

    actions1h, recommended_action1h = get_technical_voters(timeframe="1h")
    actions4h, recommended_action4h = get_technical_voters(timeframe="4h")
    actions1d, recommended_action1d = get_technical_voters(timeframe="1d")

    LSTM_long_prob, LSTM_short_prob = infer_LSTM()
    transformer_long_prob, transformer_short_prob = infer_Transformer()

    context = {
        # first column
        "aipha_long_prob_15m": aipha_predictions15m[0] * 100,
        "aipha_short_prob_15m": aipha_predictions15m[1] * 100,
        "aipha_date_obj15m": aipha_date_obj15m,
        "aipha_date_obj_end15m": aipha_date_obj_end15m,
        "aipha_enter_price15m": aipha_enter_price15m,

        "aiphabot_long_prob_1h": aipha_predictions1h[0] * 100,
        "aiphabot_short_prob_1h": aipha_predictions1h[1] * 100,
        "aiphabot_hold_prob_1h": aipha_predictions1h[2] * 100,
        "aiphabot_date_obj1h": aipha_date_obj1h,
        "aipha_date_obj_end1h": aipha_date_obj_end1h,
        "aipha_enter_price1h": aipha_enter_price1h,
        "aipha_long_take_profit1h": aipha_long_take_profit1h,
        "aipha_short_take_profit1h": aipha_short_take_profit1h,

        "aiphabot_long_prob_4h": aipha_predictions4h[0] * 100,
        "aiphabot_short_prob_4h": aipha_predictions4h[1] * 100,
        "aiphabot_hold_prob_4h": aipha_predictions4h[2] * 100,
        "aiphabot_date_obj4h": aipha_date_obj4h,
        "aipha_date_obj_end4h": aipha_date_obj_end4h,
        "aipha_enter_price4h": aipha_enter_price4h,
        "aipha_long_take_profit4h": aipha_long_take_profit4h,
        "aipha_short_take_profit4h": aipha_short_take_profit4h,

        "aiphabot_long_prob_1d": aipha_predictions1d[0] * 100,
        "aiphabot_short_prob_1d": aipha_predictions1d[1] * 100,
        "aiphabot_hold_prob_1d": aipha_predictions1d[2] * 100,
        "aipha_date_obj1d": aipha_date_obj1d,
        "aipha_date_obj_end1d": aipha_date_obj_end1d,
        "aipha_enter_price1d": aipha_enter_price1d,
        "aipha_long_take_profit1d": aipha_long_take_profit1d,
        "aipha_short_take_profit1d": aipha_short_take_profit1d,

        "technical_actions_1h": actions1h,
        "technical_recommended_action_1h": recommended_action1h,
        "technical_actions_4h": actions4h,
        "technical_recommended_action_4h": recommended_action4h,
        "technical_actions_1d": actions1d,
        "technical_recommended_action_1d": recommended_action1d,

        "LSTM_long_prob": LSTM_long_prob,
        "LSTM_short_prob": LSTM_short_prob,

        "transformer_long_prob": transformer_long_prob,
        "transformer_short_prob": transformer_short_prob,

        # second column
        "data_fng": data_fng,
        'scraped_data': scraped_data,
        'avg_sentiment_scores': avg_sentiment_scores_percentage,
        'sentiment_labels': sentiment_labels,

        # third column
        "data_global": data_global,
        "kimchi_data": kimchi_data,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'kendall_corr': kendall_corr,
    }

    return render(request, 'independent_indicator_views.html', context)
