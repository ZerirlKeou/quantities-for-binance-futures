from binance.um_futures import UMFutures
import pandas as pd
import json

client = UMFutures()


def future_pool(info=client.exchange_info()):
    info = pd.DataFrame(info['symbols'])
    pair = dict(zip(info.symbol.values, info.pair.values))
    with open("future_pool.json", "w", encoding='utf-8') as f:
        json.dump(pair, f, ensure_ascii=False, indent=4)


future_pool()
