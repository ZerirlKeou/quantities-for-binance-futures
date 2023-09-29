from grid_trader import grid_static
import pandas as pd


class GridTradeBack(object):
    def __init__(self, money, premium, interval, lever):
        self.original_money = money
        self.money = money
        self.lever = lever
        self.premium = premium
        self.interval = interval
        self.df = self._read_clean_data()
        self._profit(self.df)

    def _read_clean_data(self):
        self.path = str(self.interval) + "_data.csv"
        df = pd.read_csv(self.path)
        df['change'] = df['Return_1'].shift(1)
        self.gridStack = grid_static.InitGrid(close=df.loc[0, "Close"], low_price=222.45, high_price=350, size=100)
        return df

    def _profit(self, df):
        buyStack = self.gridStack.buy_stack
        sellStack = self.gridStack.sell_stack
        right_now_blank_price = self.gridStack.middle_price
        heldbuyStack = []
        heldsellStack = []
        df['Profit'] = 0.0
        print(buyStack)
        print(sellStack)
        direction = "none"
        for i in range(len(df)):
            print("网格做多开仓价格", buyStack[-1:][0])
            print("网格做空开仓价格", sellStack[0])
            print("当前收盘价", df.loc[i, "Close"])
            # 做多
            if direction == "none":
                if df.loc[i, "Close"] <= buyStack[-1:][0]:
                    """push to held stack and pop pre stack"""
                    result = [num for num in buyStack if num > df.loc[i, "Close"]]
                    print("做多的结果", result)
                    for res in result:
                        heldbuyStack.append([res, right_now_blank_price])
                    print("当前做多持仓", heldbuyStack)
                    # 网格后退
                    buyStack = buyStack[:-len(result)]
                    # sellStack.insert(0, right_now_blank_price)
                    direction = "more"
                    print("当前做多等待配对", right_now_blank_price)
                    right_now_blank_price = buyStack[-1:][0]
                # 做空
                elif df.loc[i, "Close"] >= sellStack[0]:
                    """push to held stack and pop pre stack"""
                    result = [num for num in sellStack if num < df.loc[i, "Close"]]
                    print("做空的结果", result)
                    for res in result:
                        heldsellStack.append(res)
                    print("当前做空持仓", heldsellStack)
                    sellStack = sellStack[:-1]
                    buyStack.append(right_now_blank_price)
                    right_now_blank_price = sellStack[-1:][0]
                    # df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                    # self.money += df.loc[i, 'Profit']
            elif direction == "more":
                if df.loc[i, "Close"] > heldbuyStack[0][1]:
                    df.loc[i, 'Profit'] = heldbuyStack[0][1] - heldbuyStack[0][0]
                else:
                    pass
            else:
                if df.loc[i, "Close"] < heldsellStack[0][1]:
                    df.loc[i, 'Profit'] = heldsellStack[0][1] - heldsellStack[0][0]

    def _draw_html(self):
        pass


def test_grid_trade():
    grid = GridTradeBack(48, 0.002, "5m", 15)
