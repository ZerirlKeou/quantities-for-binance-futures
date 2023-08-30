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
        heldbuyStack = []
        heldsellStack = []
        df['Profit'] = 0.0
        for i in range(len(df)):
            print(buyStack[-1:][0])
            if df.loc[i, "Close"] <= buyStack[-1:][0]:
                """push to held stack and pop pre stack"""
                result = [num for num in buyStack if num > df.loc[i, "Close"]]
                print(result)
                heldbuyStack.append(result)
                print(heldsellStack)
                # buyStack = buyStack[:-len(result)]
                df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                self.money += df.loc[i, 'Profit']
            else:
                result = [num for num in buyStack if num > df.loc[i, "Close"]]
            #     elif df.loc[i, "Close"] >= sellStack[0]:
            #         bought = True
            #         situation = 'less'
            #         df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
            #         self.money += df.loc[i, 'Profit']
            # else:
            #     if situation == 'more':
            #         if df.loc[i, "Close"] == -1:
            #             situation = "less"
            #             df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * df.loc[
            #                 i, 'change'] * self.lever
            #             self.money += df.loc[i, 'Profit']
            #         else:
            #             df.loc[i, 'Profit'] += self.money * df.loc[i, 'change'] * self.lever
            #             if df.loc[i, 'Profit'] <= -self.money * stop_loss_percentage:
            #                 df.loc[i, 'Profit'] = -self.money * stop_loss_percentage  # 卖出，设置亏损10.2%
            #                 self.money += df.loc[i, 'Profit']
            #                 bought = False
            #             else:
            #                 self.money += df.loc[i, 'Profit']
            #     elif situation == 'less':
            #         if df.loc[i, "Close"] == 1:
            #             situation = "more"
            #             df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * df.loc[
            #                 i, 'change'] * self.lever
            #             self.money += df.loc[i, 'Profit']
            #         else:
            #             df.loc[i, 'Profit'] += -self.money * df.loc[i, 'change'] * self.lever
            #             if df.loc[i, 'Profit'] <= -self.money * stop_loss_percentage:
            #                 df.loc[i, 'Profit'] = -self.money * stop_loss_percentage  # 卖出，设置亏损10.2%
            #                 self.money += df.loc[i, 'Profit']
            #                 bought = False
            #             else:
            #                 self.money += df.loc[i, 'Profit']

    def _draw_html(self):
        pass


def test_grid_trade():
    grid = GridTradeBack(48, 0.002, "5m", 15)
