import numpy as np


class FactorCalculator:
    def __init__(self):
        pass

    def calculate_williams_r(self, df, period=14):
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        df['williams_r'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
        return df

    def calculate_macd(self, df, short_period=12, long_period=26, signal_period=9):
        short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
        df['dif'] = short_ema - long_ema
        df['dea'] = df['dif'].ewm(span=signal_period, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        return df

    def calculate_cci(self, df, window=14):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        mean_price = typical_price.rolling(window=window).mean()
        mean_abs_deviation = np.abs(typical_price - mean_price).rolling(window=window).mean()
        df['CCI'] = (typical_price - mean_price) / (0.015 * mean_abs_deviation)
        return df

    def calculate_volume(self, df):
        volume_mean = df['Volume'].ewm(span=5, adjust=False).mean()
        df['volume_change'] = df['Volume'] / volume_mean
        return df

    def calculate_factors(self, df):
        df['Return_1'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
        df['Return_2'] = (df['Close'].shift(-2) - df['Close']) / df['Close']
        df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
        df['Return_4'] = (df['Close'].shift(-4) - df['Close']) / df['Close']
        df['Return_5'] = (df['Close'].shift(-5) - df['Close']) / df['Close']
        df = self.calculate_williams_r(df)
        df = self.calculate_macd(df)
        df = self.calculate_cci(df)
        df = self.calculate_volume(df)
        return df
