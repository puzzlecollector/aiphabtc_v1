import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from django.shortcuts import render
from django.shortcuts import render
import requests
import csv
from io import StringIO
from datetime import datetime, timedelta
import openai
from django.http import JsonResponse
import time
import yfinance as yf
import ccxt
import pyupbit
from django.conf import settings
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.tsa.arima.model import ARIMA
from pytz import timezone
from prophet import Prophet
import pickle
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timezone
import zoneinfo
from zoneinfo import ZoneInfo  # Import ZoneInfo
import json
import faiss
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tokenization_roberta_spm import FairSeqRobertaSentencePieceTokenizer
import pickle
import torch
import plotly.graph_objects as go
from plotly.io import to_html
import plotly.io as pio
from plotly.graph_objs import Scatter
import plotly.express as px


def granger_causality_test(data, max_lag):
    test = 'ssr_chi2test'
    result = grangercausalitytests(data, max_lag, verbose=True)
    p_values = [round(result[i + 1][0][test][1], 4) for i in range(max_lag)]
    return p_values


def align_lengths(x, y):
    max_len = max(len(x), len(y))
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    x_filled = x_series.reindex(range(max_len), method='ffill').to_numpy()
    y_filled = y_series.reindex(range(max_len), method='ffill').to_numpy()
    return x_filled, y_filled


def get_correlations(x, y, max_lags=5):
    x_aligned, y_aligned = align_lengths(x, y)
    pearson_corr, _ = pearsonr(x_aligned, y_aligned)
    spearman_corr, _ = spearmanr(x_aligned, y_aligned)
    x_ = np.array(x_aligned).reshape(-1, 1)
    y_ = np.array(y_aligned).reshape(-1, 1)
    data = np.hstack((x_, y_))
    granger_p_values = granger_causality_test(data, max_lags)
    coint_test = coint(x_, y_)
    coint_p_values = coint_test[1]
    data = {}
    data["pearson"] = pearson_corr
    data["spearman"] = spearman_corr
    data["granger"] = granger_p_values
    data["coint"] = coint_p_values
    return data


def indicator_view(request):
    # btc, eth
    bitget = ccxt.bitget()
    btcusdt = bitget.fetch_ohlcv("BTC/USDT:USDT", "1d")
    ethusdt = bitget.fetch_ohlcv("ETH/USDT:USDT", "1d")
    btc = pd.DataFrame(btcusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth = pd.DataFrame(ethusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_close = btc.close.values
    eth_close = eth.close.values
    btc_eth_data = get_correlations(btc_close, eth_close)

    # btc, gold
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)
    gold_data = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
                            interval="1d")
    gold_close = gold_data["Close"].values[-200:]
    btc_gold_data = get_correlations(btc_close, gold_close)

    context = {
        'btc_eth_data': btc_eth_data,
        'btc_gold_data': btc_gold_data,
        'corr_chat_message': '',
    }

    return render(request, 'ai_indicator_page.html', context)


def fetch_ai_corr(request):
    # btc, eth
    bitget = ccxt.bitget()
    btcusdt = bitget.fetch_ohlcv("BTC/USDT:USDT", "1d")
    ethusdt = bitget.fetch_ohlcv("ETH/USDT:USDT", "1d")
    btc = pd.DataFrame(btcusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth = pd.DataFrame(ethusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_close = btc.close.values
    eth_close = eth.close.values
    btc_eth_data = get_correlations(btc_close, eth_close)

    # btc, gold
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)
    gold_data = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
                            interval="1d")
    gold_close = gold_data["Close"].values[-200:]
    btc_gold_data = get_correlations(btc_close, gold_close)

    message = ("비트코인과 이더리움의 상관관계 정보야. 정확히는 피어슨 상관계수, 스피어만 상관계수, 그레인저 인과성 검정 (다양한 lag에 대한 p-값), 그리고 공적분 검정 p-값이야:\n"
               "{}\n"
               "비트코인과 금의 상관관계 정보야:\n"
               "{}\n"
               "이 관계를 해석해줘.").format(str(btc_eth_data), str(btc_gold_data))
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


def get_predictions_fbprophet(btc_sequence, timestamps):
    df = pd.DataFrame({
        'ds': timestamps,
        'y': btc_sequence,
    })
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=6, freq='30min')
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


def preprocess(df):
    bitget = ccxt.bitget()
    dates = df["timestamp"].values
    timestamps = []
    korean_timezone = zoneinfo.ZoneInfo("Asia/Seoul")
    for i in range(len(dates)):
        date_string = bitget.iso8601(int(dates[i]))
        date_object = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")
        korean_time = date_object.astimezone(korean_timezone)
        formatted_korean_time = korean_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(formatted_korean_time)
    df["datetime"] = timestamps
    df = df.drop(columns={"timestamp"})
    return df


def time_series_views(request):
    bitget = ccxt.bitget()
    btcusdt = bitget.fetch_ohlcv("BTC/USDT:USDT", "30m")
    btc = pd.DataFrame(btcusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_close = btc['close'].values

    labels = [
        datetime.utcfromtimestamp(t / 1000)
        .replace(tzinfo=timezone.utc)
        .astimezone(ZoneInfo('Asia/Seoul'))  # Use ZoneInfo instead of timezone
        .strftime('%Y-%m-%d %H:%M:%S')
        for t in btc['timestamp']
    ]

    # Get predictions
    prophet_forecast = get_predictions_fbprophet(btc_close, labels)
    arima_forecast = get_predictions_arima(btc_close)

    # Convert numpy float32 to Python float for JSON serialization
    prophet_forecast = [float(f) for f in prophet_forecast]
    arima_forecast = [float(f) for f in arima_forecast]

    # Append labels for forecast
    last_label = labels[-1]
    last_timestamp = datetime.strptime(last_label, '%Y-%m-%d %H:%M:%S')
    interval = timedelta(minutes=30)

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


### news pattern matching tool ###
# load necessary files
tokenizer = FairSeqRobertaSentencePieceTokenizer.from_pretrained("pybo/views/fairseq-roberta-all-model")

embedding_model = AutoModel.from_pretrained("pybo/views/cryptoroberta")
config = AutoConfig.from_pretrained("pybo/views/cryptoroberta")

index = faiss.read_index('pybo/views/roberta_index.faiss')
published_datetimes = np.load("pybo/views/published_datetimes.npy", allow_pickle=True)
# read candidate texts
with open('pybo/views/candidate_texts.pkl', 'rb') as f:
    candidate_texts = pickle.load(f)


def get_query_embedding(query):
    encoded_query = tokenizer(query, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    embedding_model.eval()
    with torch.no_grad():
        query_embedding = embedding_model(**encoded_query)[0][:, 0, :]
        query_embedding = query_embedding.numpy()
    return query_embedding


def convert_json_chart_data_to_pd(json_file: str):
    with open(json_file) as f:
        d = json.load(f)
    chart_df = pd.DataFrame(d)
    chart_df = chart_df.rename(columns={0: "timestamp", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"})
    binance = ccxt.binance()
    dates = chart_df["timestamp"].values
    timestamp = []
    for i in range(len(dates)):
        date_string = binance.iso8601(int(dates[i]))
        date_string = date_string[:10] + " " + date_string[11:-5]
        timestamp.append(date_string)
    chart_df["datetime"] = timestamp
    chart_df.drop(columns={"timestamp"}, inplace=True)
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)
    return chart_df


def get_relevant_chart_segment(chart_df, datestr):
    df30m_idx = -1
    cur_date = chart_df["datetime"].values
    news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    for i in range(len(cur_date) - 1):
        current_date = datetime.strptime(cur_date[i], "%Y-%m-%d %H:%M:%S")
        next_date = datetime.strptime(cur_date[i + 1], "%Y-%m-%d %H:%M:%S")
        if news_datestr >= current_date and news_datestr < next_date:
            df30m_idx = i
            break
    return df30m_idx


# read chart data
df30m = convert_json_chart_data_to_pd("pybo/views/BTC_USDT-30m-4.json")


def inner_product_to_percentage(inner_product):
    return (inner_product + 1) / 2 * 100

def search_news(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("news_text")
        # Call the functions to perform the search
        query_embedding = get_query_embedding(query)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, 1000)
        topk = 5
        results = []
        for i in range(topk):
            text = candidate_texts[indices[0][i]]
            text = text.replace('\n', '<br>')
            similarity = round(inner_product_to_percentage(distances[0][i]), 3)
            date = published_datetimes[indices[0][i]]
            relevant_chart_idx = get_relevant_chart_segment(df30m, date)
            relevant_chart_segment = df30m.iloc[
                                     relevant_chart_idx:relevant_chart_idx + 48]  # relevant chart data for the next 24 hours

            # Create data for the chart
            chart_data = {
                'x': relevant_chart_segment.index.tolist(),
                'y': relevant_chart_segment['close'].tolist(),
            }

            # Add data for each result
            results.append({
                'text': text,
                'similarity': similarity,
                'date': date,
                'chart_data': chart_data,
            })

        # Return the results as JSON
        return JsonResponse({'results': results})
