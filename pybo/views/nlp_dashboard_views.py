from django.shortcuts import render
import sys
import os
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
import plotly.graph_objects as go # plotly needed for chart plotting
from plotly.io import to_html
import plotly.io as pio
from plotly.graph_objs import Scatter
import plotly.express as px


# news similarity indicator
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


def get_relevant_chart_segment1d(chart_df, datestr):
    chart_df.set_index(pd.DatetimeIndex(chart_df["date"]), inplace=True)
    df1d_idx = -1
    cur_date = chart_df["date"].values  # Ensure this column contains date and time
    news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    for i in range(len(cur_date) - 1):
        # Convert numpy.datetime64 to string and then to datetime
        current_date_str = cur_date[i].astype(str)
        current_date_str = current_date_str.split('T')[0] + ' ' + current_date_str.split('T')[1].split('.')[0]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S")
        next_date_str = cur_date[i + 1].astype(str)
        next_date_str = next_date_str.split('T')[0] + ' ' + next_date_str.split('T')[1].split('.')[0]
        next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S")
        if news_datestr >= current_date and news_datestr < next_date:
            df1d_idx = i
            break
    return df1d_idx


# read chart data
df30m = convert_json_chart_data_to_pd("pybo/views/BTC_USDT-30m-4.json")

# read 1d chart data
df1d = pd.read_feather("pybo/views/BTC_USDT-1d_.feather")  # !pip install pyarrow


def inner_product_to_percentage(inner_product):
    return (inner_product + 1) / 2 * 100

def search_news(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("news_text")
        topk = int(data.get("top_k", 5))
        topk = max(5, min(topk, 20))
        # Call the functions to perform the search
        query_embedding = get_query_embedding(query)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, 1000)
        results = []
        for i in range(topk):
            text = candidate_texts[indices[0][i]]
            text = text.replace('\n', '<br>')
            similarity = round(inner_product_to_percentage(distances[0][i]), 3)
            date = published_datetimes[indices[0][i]]
            relevant_chart_idx_30m = get_relevant_chart_segment(df30m, date)
            relevant_chart_segment_30m = df30m.iloc[
                                         relevant_chart_idx_30m:relevant_chart_idx_30m + 48]  # relevant chart data for the next 24 hours


            relevant_chart_idx_1d = get_relevant_chart_segment1d(df1d, date)
            relevant_chart_segment_1d = df1d.iloc[relevant_chart_idx_1d:relevant_chart_idx_1d + 30] # relevant chart data for the next 1 month

            # Create data for the chart
            chart_data_30m = {
                'x': relevant_chart_segment_30m.index.tolist(),
                'y': relevant_chart_segment_30m['close'].tolist(),
            }

            chart_data_1d = {
                'x': relevant_chart_segment_1d.index.tolist(),
                'y': relevant_chart_segment_1d['close'].tolist(),
            }

            # Add data for each result
            results.append({
                'text': text,
                'similarity': similarity,
                'date': date,
                'chart_data_30m': chart_data_30m,
                'chart_data_1d': chart_data_1d,
            })

        # Return the results as JSON
        return JsonResponse({'results': results})

def nlp_views(request):
    return render(request, 'nlp_dashboard.html')

