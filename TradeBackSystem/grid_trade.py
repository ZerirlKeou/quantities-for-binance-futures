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
        self.direction = "none"
        self.right_now_blank_price = self.gridStack.middle_price
        self.buyStack = self.gridStack.buy_stack
        self.holdBuyStack = []
        self.holdSellStack = []
        self._profit(self.df)

    def _read_clean_data(self):
        self.path = str(self.interval) + "_data.csv"
        df = pd.read_csv(self.path)
        df['change'] = df['Return_1'].shift(1)
        self.gridStack = grid_static.InitGrid(close=df.loc[0, "Close"], low_price=222.45, high_price=350, size=100)
        return df

    def _buy_grid(self, df, i):
        self.direction = "more"
        """push to held stack and pop pre stack"""
        result = [num for num in self.buyStack if num > df.loc[i, "Close"]]
        print("做多的结果", result)
        for res in result:
            self.holdBuyStack.append([res, self.right_now_blank_price])
        print("当前做多持仓", self.holdBuyStack)
        # 网格后退
        self.right_now_blank_price = self.buyStack[-len(result):][0]
        self.buyStack = self.buyStack[:-len(result)]

    def _profit(self, df):
        sellStack = self.gridStack.sell_stack
        df['Profit'] = 0.0
        print(self.buyStack)
        print(sellStack)
        for i in range(len(df)):
            print("当前为第", i + 1, "天")
            print("网格做多开仓价格", self.buyStack[-1:][0])
            print("网格做空开仓价格", sellStack[0])
            print("当前收盘价", df.loc[i, "Close"])
            # 做多
            if self.direction == "none":
                if df.loc[i, "Close"] <= self.buyStack[-1:][0]:
                    self._buy_grid(df=df, i=i)
                # 做空
                elif df.loc[i, "Close"] >= sellStack[0]:
                    self.direction = "less"
                    """push to held stack and pop pre stack"""
                    result = [num for num in sellStack if num < df.loc[i, "Close"]]
                    print("做空的结果", result)
                    for res in result:
                        self.holdSellStack.append(res)
                    print("当前做空持仓", self.holdSellStack)
                    sellStack = sellStack[:-1]
                    self.buyStack.append(self.right_now_blank_price)
                    self.right_now_blank_price = sellStack[-1:][0]
            elif self.direction == "more":
                if df.loc[i, "Close"] > self.holdBuyStack[-1][1]:
                    tempStack = []
                    for num in reversed(self.holdBuyStack):
                        if num[1] > df.loc[i, "Close"]:
                            break
                        if num[1] < df.loc[i, "Close"]:
                            df.loc[i, 'Profit'] += num[1] - num[0]
                            tempStack.append(num)
                            self.right_now_blank_price = num[1]
                            self.holdBuyStack.pop(-1)
                            print("卖出后持仓", self.holdBuyStack)
                        else:
                            pass
                    if tempStack:
                        for ts in reversed(tempStack):
                            self.buyStack.append(ts[0])
                            print(self.buyStack)
                        self.right_now_blank_price = tempStack[-1][1]
                    if not self.holdBuyStack:
                        self.direction = "none"
                elif df.loc[i, "Close"] < self.buyStack[-1:][0]:
                    self._buy_grid(df=df, i=i)
            else:
                if df.loc[i, "Close"] < self.holdSellStack[0][1]:
                    df.loc[i, 'Profit'] = self.holdSellStack[0][1] - self.holdSellStack[0][0]
                else:
                    pass

    def _draw_html(self):
        pass


def test_grid_trade():
    GridTradeBack(48, 0.002, "5m", 15)
