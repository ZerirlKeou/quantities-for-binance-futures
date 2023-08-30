import csv
from binance.um_futures import UMFutures

client = UMFutures()

# 定义时间间隔和文件名
intervals = ['1d', '1h', '15m', '5m', '1m']
file_names = ['1d_data.csv', '1h_data.csv', '15m_data.csv', '5m_data.csv', '1m_data.csv']

for interval, file_name in zip(intervals, file_names):
    # 获取行情数据
    data = client.continuous_klines(pair="BNBUSDT",contractType='PERPETUAL', interval=interval, limit=1500)

    # 创建CSV文件并写入数据
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                         'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

        for row in data:
            writer.writerow(row)

    print(f"已创建 {file_name}")

print("行情数据池创建完成！")
