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
    def macd_time_series(self, df, T):
        data = df.copy()
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)
        resampled_data = data.resample(f'{T}T').agg({
            'Open':'first',
            'High':'max',
            'Low':'min',
            'Close':'last',
            'Volume':'sum'
        })

        short_period, long_period, signal_period = 12, 26, 9
        short_ema = resampled_data['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = resampled_data['Close'].ewm(span=long_period, adjust=False).mean()
        resampled_data['dif_high_level'] = short_ema - long_ema
        resampled_data['dea_high_level'] = resampled_data['dif_high_level'].ewm(span=signal_period, adjust=False).mean()
        resampled_data['macd_high_level'] = (resampled_data['dif_high_level'] - resampled_data['dea_high_level']) * 2
        resampled_data = resampled_data[['dif_high_level','dea_high_level','macd_high_level']]

        try:
            df.drop(['dif_high_level','dea_high_level','macd_high_level'], axis=1, inplace=True)
        except:
            pass

        resampled_data_expanded = resampled_data.reindex(resampled_data.index.repeat(T)).reset_index(drop=True)
        resampled_data_expanded['Open time'] = df['Open time']
        resampled_data_expanded.dropna(inplace=True)

        df_merged = pd.merge(df, resampled_data_expanded, on='Open time')



        df_merged['macd_back_time_series'] = np.where((df_merged['macd_back']==1) & (df_merged['macd_high_level'] < 0),1,np.where((df_merged['macd_back']==-1) & (df_merged['macd_high_level'] > 0),-1,0))


        return df_merged


class TimeSeriesFactorData(TimeSeriesFactorCalculate):
    def __init__(self):
        TimeSeriesFactorCalculate.__init__(self)
        self.df = None
        self.pool = ['macd_time_series']

    def calculate_factors(self, df):
        self.df = df
        interval =  df['Open time'][1] - df['Open time'][0]
        if interval == 60000:
            T = 5
        elif interval == 300000:
            T = 3
        elif interval == 900000:
            T = 4
        elif interval == 3600000:
            T = 24
        elif interval == 24*3600000:
            T = 30
        else:
            print('当前时间序列未找到合适的父级序列进行合并，请检查dataframe中的时间戳是否正确')
            T = 5
        for types in self.pool:
            view_function = self.dfp.route_output(types)
            self.df = view_function(self, self.df, T)
        return self.df