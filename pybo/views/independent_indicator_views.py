# Standard library imports
import sys
import os
import csv
import time
import json
import asyncio
from io import StringIO
from datetime import datetime, timedelta
from datetime import timezone  # This is duplicated with the above import, so combined them
import pickle

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
from pytz import timezone
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

        # second column
        "data_fng": data_fng,

        # third column
        "data_global": data_global,
        "kimchi_data": kimchi_data
    }

    return render(request, 'independent_indicator_views.html', context)
