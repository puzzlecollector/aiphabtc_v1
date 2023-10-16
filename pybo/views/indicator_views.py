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
        # openai.api_key = "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"
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
