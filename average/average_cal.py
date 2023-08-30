import pandas as pd


class MultiAverageLine:
    def __init__(self, interval):
        self.namelist = []
        self.L = self.Fibonacci_Sequence(time=13)
        df = self.read_data_frame(interval)
        for i in self.L:
            self.MA(df, number=i)
        df = self.clean_Data(df)
        print(self.namelist)
        print(df)

    def read_data_frame(self, interval):
        df = pd.read_csv("data\\" + str(interval) + "_data.csv")
        return df

    def clean_Data(self, df):
        df.dropna(subset=self.namelist, inplace=True)
        df = df.reset_index(drop=True)
        return df

    def Fibonacci_Sequence(self, time):
        L = [1, 2]
        while len(L) < time:
            L.append(L[-1] + L[-2])
        return L

    def MA(self, df, number):
        df['CloseMA' + str(number)] = df['Close'].rolling(number).mean()
        self.namelist.append('CloseMA' + str(number))
