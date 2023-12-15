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

    @dfp.route_types(u'')
    @dfp.route_types(u'other')
    def williams_point(self, df):
        df['williams_1'] = (-df['williams_r'].shift(1) - 80) / (-df['williams_r'] - 80)
        df['low-open'] = df['Low'] - df['Open'] / df['Close']
        df['high-open'] = df['High'] - df['Open'] / df["Close"]
        """2023.07.16目前为止最好的一个指标"""
        df['open-close'] = (df['Open'] - df['Close']) / df["Close"]
        df['CCI_HHV'] = np.where(df["CCI"] >= 0, df['CCI'].rolling(3).max() - df['CCI'],
                                 -df['CCI'].rolling(3).max() + df['CCI'])
        df["chazhi"] = df["CCI_HHV"].shift(1) - df["CCI_HHV"]
        df["high+dif"] = df["High"] + df["dif"]
        df['rsiBuy'] = np.where(df['KR']==0,1,np.where(df['KR']==100,-1,0))
        return df

class DerivedFactorData(DerivedFactorCalculate):
    def __init__(self):
        DerivedFactorCalculate.__init__(self)
        self.df = None
        self.pool = ['macd_back', 'williams', 'basic_open_point', 'other']

    def calculate_factors(self, df):
        self.df = df
        for types in self.pool:
            view_function = self.dfp.route_output(types)
            view_function(self, self.df)
        return self.df