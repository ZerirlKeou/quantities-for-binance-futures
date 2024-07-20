import numpy as np

class DerivedFactorPool:
    def __init__(self):
        self.routes = {}

    def route_types(self, type_str):
        def decorator(f):
            self.routes[type_str] = f
            return f

        return decorator

    def route_output(self, path):
        function_val = self.routes.get(path)
        if function_val:
            return function_val
        else:
            raise ValueError('Route "{}" has not been registered'.format(path))

class DerivedFactorCalculate:
    dfp = DerivedFactorPool()

    @dfp.route_types(u'macd_back')
    def macd_back(self, df):
        df['macd_condition'] = df['macd'] > 0
        df['shifted_condition1'] = df['macd'] < df['macd'].shift(1)
        df['shifted_condition2'] = df['macd'].shift(1) > df['macd'].shift(2)
        df['deaRewrite'] = df['dea'] > df['macd']
        df['difRewrite'] = df['dif'] > df['macd']

        df['macd_back'] = np.where(
            df['deaRewrite'] & df['difRewrite'] & (
                    df['macd_condition'] & df['shifted_condition1'] & df['shifted_condition2']),
            -1,
            np.where(~df['deaRewrite'] & ~df['difRewrite'] & (
                    ~df['macd_condition'] & ~df['shifted_condition1'] & ~df['shifted_condition2']),
                     1, 0))

        df['macd_sell'] = np.where((df['macd_condition'] & df['shifted_condition1'] & df['shifted_condition2']),-1,np.where((~df['macd_condition'] & ~df['shifted_condition1'] & ~df['shifted_condition2']),1, 0))

        df.drop(['macd_condition', 'shifted_condition1', 'shifted_condition2','deaRewrite','difRewrite'], axis=1, inplace=True)
        return df

    @dfp.route_types(u'williams')
    def williams_point1(self, df):
        df['williams_condition1'] = df['williams_r'].shift(1) < -80
        df['williams_condition2'] = df['williams_r'] > -80
        df['williams_condition3'] = df['williams_r'].shift(1) > -20
        df['williams_condition4'] = df['williams_r'] < -20
        df['williams_points'] = np.where(df['williams_condition1'] & df['williams_condition2'], 1,
                                         np.where(df['williams_condition3'] & df['williams_condition4'], -1, 0))
        df.drop(['williams_condition1', 'williams_condition2', 'williams_condition3','williams_condition4'], axis=1, inplace=True)
        return df

    @dfp.route_types(u'basic_open_point')
    def basic_buy_point(self, df):
        df['basic_Open_point'] = np.where((df['macd_back'] == 1) & (df['williams_points'] == 1), 1,
                                          np.where((df['macd_back'] == -1) & (df['williams_points'] == -1), -1, 0))
        return df

    @dfp.route_types(u'cci_open_point')
    def cci_open_point(self, df):
        df['ding'] = df['CCI'].rolling(window=3).max()
        df['cci_open_point'] = np.where((df['ding']==df['CCI']) & (df['CCI']<0),1,np.where((df['ding'].shift(1)==df['CCI'].shift(1)) & (df['ding']!=df['CCI']) & (df['CCI']>0),-1,0))
    @dfp.route_types(u'golden')
    def golden_point(self,df):
        # 历史新低
        df['historical_low'] = df['Low'].rolling(window=144).min()
        # 历史新高
        df['historical_high'] = df['High'].rolling(window=144).max()
        # 多空线
        df['bull_line'] = ((df['Close'] - df['historical_low'].rolling(window=2).min()) /
                           (df['historical_high'].rolling(window=2).max() - df['historical_low'].rolling(
                               window=2).min()) * 100).rolling(window=1).mean()
        df['bear_line'] = ((df['historical_high'].rolling(window=1).max() - df['Close']) /
                           (df['historical_high'].rolling(window=1).max() - df['historical_low'].rolling(
                               window=1).min()) * 100).rolling(window=1).mean()
        # BUYPILL条件
        df['buypill'] = np.where((df['historical_low'].shift(1) == df['historical_low']) &
                                 (df['bull_line'].shift(1) < df['bear_line'].shift(1)) &
                                 (df['bull_line'] > df['bear_line']), 1, 0)
        # SELLPILL条件
        df['sellpill'] = np.where((df['historical_high'].shift(1) == df['historical_high']) &
                                  (df['bull_line'].shift(1) > df['bear_line'].shift(1)) &
                                  (df['bull_line'] < df['bear_line']), -1, 0)
        df.drop(['historical_low', 'historical_high', 'bull_line', 'bear_line'], axis=1,
                inplace=True)
        return df

    @dfp.route_types(u'other')
    def williams_point(self, df):
        df['williams_1'] = (-df['williams_r'].shift(1) - 80) / (-df['williams_r'] - 80)
        df['low-open'] = df['Low'] - df['Open'] / df['Close']
        df['high-open'] = df['High'] - df['Open'] / df["Close"]
        """2023.07.16目前为止最好的一个指标"""
        df['open-close'] = (df['Open'] - df['Close']) / df["Close"]
        df["high+dif"] = df["High"] + df["dif"]
        df['rsiBuy'] = np.where(df['KR']==0,1,np.where(df['KR']==100,-1,0))
        df['basic_open_plus'] = np.where((df['KR']==0) & (df['basic_Open_point'] == 1),1,np.where((df['KR']==100) & (df['basic_Open_point']==-1),-1,0))
        return df

    @dfp.route_types(u'macd_time_series')
    def macd_time_series(self, df):
        short_period, long_period, signal_period = 12, 26, 9
        short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
        df['dif_high_level'] = short_ema - long_ema
        df['dea_high_level'] = df['dif'].ewm(span=signal_period, adjust=False).mean()
        df['macd_high_level'] = (df['dif'] - df['dea']) * 2


class DerivedFactorData(DerivedFactorCalculate):
    def __init__(self):
        DerivedFactorCalculate.__init__(self)
        self.df = None
        self.pool = ['macd_back', 'williams', 'basic_open_point', 'cci_open_point', 'golden', 'other']

    def calculate_factors(self, df):
        self.df = df
        for types in self.pool:
            view_function = self.dfp.route_output(types)
            view_function(self, self.df)
        return self.df