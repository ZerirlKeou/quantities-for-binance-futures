import numpy as np
import pandas as pd


class TimeSeriesFactorPool:
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

class TimeSeriesFactorCalculate:
    dfp = TimeSeriesFactorPool()

    @dfp.route_types(u'macd_time_series')
    def macd_time_series(self, df):
        print(df)
        df['Open time'] = pd.to_datetime(df['Open time'])
        df.set_index('Open time', inplace=True)
        print(df)
        # resampled_data = df.resample('5T')
        # short_period, long_period, signal_period = 12, 26, 9
        # short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
        # long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
        # df['dif_high_level'] = short_ema - long_ema
        # df['dea_high_level'] = df['dif'].ewm(span=signal_period, adjust=False).mean()
        # df['macd_high_level'] = (df['dif'] - df['dea']) * 2
        return df


class TimeSeriesFactorData(TimeSeriesFactorCalculate):
    def __init__(self):
        TimeSeriesFactorCalculate.__init__(self)
        self.df = None
        self.pool = ['macd_time_series']

    def calculate_factors(self, df):
        self.df = df
        for types in self.pool:
            view_function = self.dfp.route_output(types)
            view_function(self, self.df)
        return self.df