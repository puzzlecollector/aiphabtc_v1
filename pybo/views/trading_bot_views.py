from django.shortcuts import render
import ccxt
import zoneinfo
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pandas_ta as ta
from django.conf import settings
import os


from datetime import datetime, timezone
import zoneinfo

def preprocess(df):
    dates = df["timestamp"].values
    timestamps = []
    korean_timezone = zoneinfo.ZoneInfo("Asia/Seoul")
    for timestamp in dates:
        date_object = datetime.utcfromtimestamp(timestamp / 1000).replace(tzinfo=timezone.utc)
        korean_time = date_object.astimezone(korean_timezone)
        formatted_korean_time = korean_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(formatted_korean_time)
    df["datetime"] = timestamps
    df = df.drop(columns={"timestamp"})
    # DEBUG: print(df["datetime"].values[-10:])
    return df

# return prediction (probability)
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

    chart_df.drop(columns={"open", "high", "low", "close", "volume", "vwap", "hwma", "datetime"}, inplace=True)
    chart_df.dropna(inplace=True)
    date_obj = chart_df.index[-1]
    date_obj_end = date_obj + pd.Timedelta(hours=4)
    date_obj = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    date_obj_end = date_obj_end.strftime('%Y-%m-%d %H:%M:%S')
    # make prediction
    model = XGBClassifier()
    model.load_model("pybo/views/xgb_optuna_optimized_4hrs")
    x = chart_df.iloc[-2, :].values.reshape((-1, chart_df.shape[1]))  # get previous timestep information
    pred = model.predict_proba(x)
    return date_obj, date_obj_end, pred[0]


def bot_view(request):
    date_obj, date_obj_end, prediction = basic_aiphabot_4hrs()
    context = {"long_prob": prediction[0] * 100,
               "short_prob": prediction[1] * 100,
               "hold_prob": prediction[2] * 100,
               "date_obj": date_obj,
               "date_obj_end": date_obj_end}
    return render(request, 'bot_views.html', context)
