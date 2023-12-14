import numpy as np
import pandas as pd


class FactorTypesPool:
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


class FactorCalculator:
    ftp = FactorTypesPool()

    @ftp.route_types(u'williams_r')
    def calculate_williams_r(self, df, period=14):
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        df['williams_r'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100

    @ftp.route_types(u'macd')
    def calculate_macd(self, df, short_period=12, long_period=26, signal_period=9):
        short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
        df['dif'] = short_ema - long_ema
        df['dea'] = df['dif'].ewm(span=signal_period, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2

    @ftp.route_types(u'cci')
    def calculate_cci(self, df, window=14):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        mean_price = typical_price.rolling(window=window).mean()
        mean_abs_deviation = np.abs(typical_price - mean_price).rolling(window=window).mean()
        df['CCI'] = (typical_price - mean_price) / (0.015 * mean_abs_deviation)

    @ftp.route_types(u'volume')
    def calculate_volume(self, df):
        volume_mean = df['Volume'].ewm(span=5, adjust=False).mean()
        df['volume_change'] = df['Volume'] / volume_mean

    @ftp.route_types(u'zjfz')
    def calculate_zjfz(self, df):
        VAR2 = df['Low'].shift(1)
        VAR3 = (abs(df['Low'] - VAR2).rolling(window=3, min_periods=1).mean() /
                (df['Low'] - VAR2).clip(lower=0).rolling(window=3, min_periods=1).mean() * 100)
        VAR4 = VAR3 * 10.0
        VAR4 = VAR4.ewm(span=3, adjust=False).mean()
        VAR5 = df['Low'].rolling(window=13).min()
        VAR6 = VAR4.rolling(window=13).max()
        VAR7 = pd.Series(np.where(df['Low'] <= VAR5, (VAR4 + VAR6 * 2) / 2, 0))
        VAR7 = VAR7.ewm(span=3, adjust=False).mean() / 618
        df['VAR8'] = np.where(VAR7 > 500, 500, VAR7)

    @ftp.route_types(u'stoch_rsi')
    def calculate_rsi_stoch_rsi(self, df, n=14, m=3):
        # 计算 LC
        df['LC'] = df['Close'].shift(1)

        # 计算 RSI
        upward_changes = np.maximum(df['Close'] - df['LC'], 0)
        downward_changes = np.abs(df['Close'] - df['LC'])
        avg_gain = upward_changes.rolling(window=n, min_periods=1).mean()
        avg_loss = downward_changes.rolling(window=n, min_periods=1).mean()
        df['RSI'] = (avg_gain / avg_loss).apply(lambda x: 100 if x == float('inf') else x)

        # 计算 Stochastic RSI
        min_RSI = df['RSI'].rolling(window=n).min()
        max_RSI = df['RSI'].rolling(window=n).max()
        df['StochRSI'] = ((df['RSI'] - min_RSI) / (max_RSI - min_RSI) * 100).fillna(0)

        # 计算 KR 和 DR
        df['KR'] = df['StochRSI'].rolling(window=m, min_periods=1).mean()
        df['DR'] = df['KR'].rolling(window=m, min_periods=1).mean()

        return df

    @ftp.route_types(u'return')
    def calculate_return(self, df):
        df['Return_1'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
        df['Return_2'] = (df['Close'].shift(-2) - df['Close']) / df['Close']
        df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
        df['Return_4'] = (df['Close'].shift(-4) - df['Close']) / df['Close']
        df['Return_5'] = (df['Close'].shift(-5) - df['Close']) / df['Close']


class WriteFactorData(FactorCalculator):
    def __init__(self):
        FactorCalculator.__init__(self)
        self.df = None
        self.pool = ['williams_r', 'macd', 'cci', 'volume', 'zjfz', 'stoch_rsi', 'return']

    def calculate_factors(self, df):
        self.df = df
        for types in self.pool:
            view_function = self.ftp.route_output(types)
            view_function(self, self.df)
        return self.df
