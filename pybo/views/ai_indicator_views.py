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

def granger_causality_test(data, max_lag):
    test = 'ssr_chi2test'
    result = grangercausalitytests(data, max_lag, verbose=True)
    p_values = [round(result[i + 1][0][test][1], 4) for i in range(max_lag)]
    return p_values

def get_correlations(x, y, max_lags=5):
    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    x_ = np.array(x).reshape(-1, 1)
    y_ = np.array(y).reshape(-1, 1)
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
    btcusdt = bitget.fetch_ohlcv("BTC/USDT", "30m")
    ethusdt = bitget.fetch_ohlcv("ETH/USDT", "30m")
    btc = pd.DataFrame(btcusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth = pd.DataFrame(ethusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_close = btc.close.values
    eth_close = eth.close.values
    btc_eth_data = get_correlations(btc_close, eth_close)

    # btc, gold
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    gold_data = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
                            interval="30m")
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
    btcusdt = bitget.fetch_ohlcv("BTC/USDT", "30m")
    ethusdt = bitget.fetch_ohlcv("ETH/USDT", "30m")
    btc = pd.DataFrame(btcusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth = pd.DataFrame(ethusdt, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_close = btc.close.values
    eth_close = eth.close.values
    btc_eth_data = get_correlations(btc_close, eth_close)

    # btc, gold
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    gold_data = yf.download("GC=F", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
                            interval="30m")
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