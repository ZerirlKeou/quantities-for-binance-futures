import pandas as pd
from binance.um_futures import UMFutures

client = UMFutures()


def binance_future(pair="BNBUSDT", start_val=1683956800000, end_val=1684956800000, interval="15m",withColumns=True, limit=1500):
    data = client.continuous_klines(pair=pair, contractType='PERPETUAL', startTime=start_val,
                                    endTime=end_val, interval=interval, limit=limit)
    if withColumns:
        columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
                   "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
        return pd.DataFrame(data, columns=columns)
    else:
        return pd.DataFrame(data)


def test():
    df = binance_future()
    print(df)
