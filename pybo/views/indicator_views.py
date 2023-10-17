from django.shortcuts import render
import requests
import csv
from io import StringIO
from datetime import datetime
import openai
from django.http import JsonResponse
import time
import yfinance as yf
import ccxt
import pyupbit
from django.conf import settings
import pandas as pd

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

def get_technical_indicators():
    bitget = ccxt.bitget()
    ohlcv = bitget.fetch_ohlcv("BTC/USDT", "30m")
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


def indicator_view(request):
    url_fng = 'https://api.alternative.me/fng/?limit=7&date_format=kr'
    response_fng = requests.get(url_fng)

    url_global = 'https://api.coinlore.net/api/global/'
    response_global = requests.get(url_global)

    if response_fng.status_code == 200 and response_global.status_code == 200:
        data_fng = response_fng.json().get('data', [])
        data_global = response_global.json()  # Assuming the API returns a list, we pick the first item
        context = {
            'data': data_fng,
            'global_data': data_global,
            'chat_message': '',  # initially set to an empty string
            'global_chat_message': '',
            'kimchi_data': get_kimchi_data(),
            'technical_data': get_technical_indicators(),
            'technical_chat_message': '',
        }
        return render(request, 'indicator_page.html', context)
    else:
        # Handle error here
        return render(request, 'indicator_page.html', {'error_message': 'Failed to fetch the data.'})

def fetch_ai_analysis(request):
    url = 'https://api.alternative.me/fng/?limit=7&date_format=kr'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('data', [])
        message = "과거 7일동안의 크립토 공포 탐욕 지수야: "
        message += ', '.join([str(item['value']) for item in data])
        message += "\n 해당 정보를 분석해서 비트코인 가격 추세에 어떤 영향을 미칠지 해석해줘."
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
    else:
        return JsonResponse({'error_message': 'Failed to fetch the data.'})

def fetch_ai_analysis_global(request):
    url_global = 'https://api.coinlore.net/api/global/'
    response_global = requests.get(url_global)

    if response_global.status_code == 200:
        data = response_global.json()[0]  # assuming the API returns a list, we pick the first item

        message = ("다음과 같은 글로벌 크립토 시장 정보가 주어졌을때 이 정보들을 해석하고 크립토 시장의 미래에 대해서 분석해줘 "
                   "Coins Count: {coins_count}, Active Markets: {active_markets}, "
                   "Total Market Cap ($): {total_mcap}, Total Volume: {total_volume}, "
                   "BTC Dominance (%): {btc_d}, ETH Dominance (%): {eth_d}, "
                   "Market Cap Change: {mcap_change}, Volume Change: {volume_change}"
                   ).format(
            coins_count=data['coins_count'],
            active_markets=data['active_markets'],
            total_mcap=data['total_mcap'],
            total_volume=data['total_volume'],
            btc_d=data['btc_d'],
            eth_d=data['eth_d'],
            mcap_change=data['mcap_change'],
            volume_change=data['volume_change']
        )

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
    else:
        return JsonResponse({'error_message': 'Failed to fetch the global crypto data.'})

def fetch_ai_technical(request):
    technical_data = get_technical_indicators()
    technical_output = technical_data["output_str"]

    # Craft the message
    message = ("다음과 같은 30분봉 비트코인 데이터가 주어졌을때:\n\n"
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

