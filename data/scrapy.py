import requests
import csv
import time
import random

def save_market_data(start_timestamp, end_timestamp, interval):
    url = "https://www.binance.com/fapi/v1/continuousKlines"
    limit = 1000
    pair = "BNBUSDT"
    contract_type = "PERPETUAL"

    if interval == "15m":
        filename = "15m_data.csv"
    elif interval == "5m":
        filename = "5m_data.csv"
    elif interval == "1m":
        filename = "1m_data.csv"
    else:
        raise ValueError("Invalid interval. Please choose from '15m', '5m', or '1m'.")

    headers = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
               "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while start_timestamp < end_timestamp:
            params = {
                "startTime": str(start_timestamp),
                "limit": str(limit),
                "pair": pair,
                "contractType": contract_type,
                "interval": interval
            }
            response = requests.get(url, params=params)
            data = response.json()

            for item in data:
                row = [item[0]]
                row.extend(item[1:12])
                writer.writerow(row)

            start_timestamp = int(data[-1][0]) + 1

            time.sleep(1 + random.random())

    print("价格信息已保存到", filename)

# 示例调用
start_timestamp = 1683956800000  # 起始时间戳
end_timestamp = int(time.time() * 1000)  # 当前时间戳

try:
    save_market_data(start_timestamp, end_timestamp, "15m")# 保存15m数据
except:
    pass
try:
    save_market_data(start_timestamp, end_timestamp, "5m")   # 保存5m数据
except:
    pass
try:
    save_market_data(start_timestamp, end_timestamp, "1m")   # 保存1m数据
except:
    pass
