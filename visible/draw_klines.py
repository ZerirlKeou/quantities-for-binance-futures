import datetime
import mpl_finance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import mpld3
import os
import plotly.graph_objects as go
import plotly.express as px


# class StockChart:
#     def __init__(self, path, name1, list, themeplot=False, draw_number=-500):
#         self.path = path
#         self.name1 = name1
#         self.list = list
#         self.theme = themeplot
#         self.draw_number = draw_number
#         self.colorPool = ['r', 'g', 'b']
#         self.df = None
#         self.figure = None
#
#     def _read_clean_data(self):
#         path = self.path
#         df = pd.read_csv(path, parse_dates=['Open time'])
#         if self.draw_number < 0:
#             df = df.iloc[self.draw_number:]
#         else:
#             df = df.iloc[:self.draw_number]
#         df = df.sort_values(by='Open time', ascending=True)
#         df['dates'] = df['Open time']
#         return df
#
#     def _plot_candlestick(self):
#         fig = go.Figure(data=[go.Candlestick(x=self.df['dates'],
#                                              open=self.df['Open'],
#                                              high=self.df['High'],
#                                              low=self.df['Low'],
#                                              close=self.df['Close'])])
#         fig.update_layout(xaxis_rangeslider_visible=True)
#         fig.update_xaxes(type='date')
#
#         return fig
#
#     def show_chart(self):
#         if self.df is None:
#             self.df = self._read_clean_data()
#
#         self.figure = self._plot_candlestick()
#
#         # Save the chart as an HTML file
#         html_file = 'Html/stock_chart.html'
#         self.figure.write_html(html_file)
#
#         return html_file


class StockChart:
    def __init__(self, path, name1, list, themeplot=False, draw_number=-500):
        self.path = path
        self.name1 = name1
        self.list = list
        self.theme = themeplot
        self.draw_number = draw_number
        self.colorPool = ['r', 'g', 'b']
        self.df = None
        self.figure = None
        self.axes = []

    def _read_clean_data(self):
        path = self.path
        df = pd.read_csv(path)
        if self.draw_number < 0:
            df = df.iloc[self.draw_number:]
        else:
            df = df.iloc[:self.draw_number]
        df = df.sort_values(by='Open time', ascending=True)
        df['dates'] = np.arange(0, len(df))
        df['trade_date2'] = [p / 1000 for p in df['Open time'].copy()]
        return df

    def _plot_candlestick(self):
        mpl_finance.candlestick_ochl(
            ax=self.axes[0],
            quotes=self.df[['dates', 'Open', 'Close', 'High', 'Low']].values,
            width=0.7,
            colorup='r',
            colordown='g',
            alpha=0.7)

        date_tickers = [datetime.datetime.fromtimestamp(t).strftime("%Y-%m") for t in self.df['trade_date2'].values]

        def format_date(x, pos):
            if x < 0 or x > len(date_tickers) - 1:
                return ''
            return date_tickers[int(x)]

        self.axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

    def format_date(self, x, pos):
        date_tickers = [datetime.datetime.fromtimestamp(t).strftime("%Y-%m") for t in self.df['trade_date2'].values]
        if x < 0 or x > len(date_tickers) - 1:
            return ''
        return date_tickers[int(x)]

    def show_chart(self):
        if self.df is None:
            self.df = self._read_clean_data()

        if self.figure is None:
            self.figure = plt.figure(figsize=(12, 9), facecolor='#ffffff')

        plt.show()


class StockChart2(StockChart):
    def __init__(self, path, name1, list, themeplot=False, draw_number=-500):
        super().__init__(path, name1, list, themeplot, draw_number)
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ma_data = {}

    def _calculate_ma(self):
        self.ma_data['5'] = self.df.Close.rolling(5).mean()
        self.ma_data['8'] = self.df.Close.rolling(20).mean()
        self.ma_data['13'] = self.df.Close.rolling(30).mean()
        self.ma_data['21'] = self.df.Close.rolling(60).mean()
        self.ma_data['55'] = self.df.Close.rolling(120).mean()
        self.ma_data['144'] = self.df.Close.rolling(250).mean()

    def _plot_candlestick(self):
        super()._plot_candlestick()

        for ma in ['5', '8', '13', '21', '55', '144']:
            self.axes[0].plot(self.df['dates'], self.ma_data[ma], label=ma)

    def _plot_custom_subplot1(self):
        self.axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(self.format_date))
        if self.name1 == "williams_r":
            self.axes[1].plot(self.df['dates'], self.df["williams_r"], color='b', alpha=0.7)
            self.axes[1].set_ylabel('williams', color='w')
        elif self.name1 == 'CCI':
            self.axes[1].plot(self.df['dates'], self.df["CCI"], color='b', alpha=0.7)
            self.axes[1].plot(self.df['dates'], self.df["CCI_HHV"], color='r', alpha=0.7)
            self.axes[1].set_ylabel('CCI', color='w')
        else:
            self.df['up'] = self.df.apply(lambda row: 1 if row['macd'] > 0 else 0, axis=1)
            self.axes[1].bar(self.df.query('up == 1')['dates'], self.df.query('up == 1')['macd'], color='r', alpha=0.7)
            self.axes[1].bar(self.df.query('up == 0')['dates'], self.df.query('up == 0')['macd'], color='g', alpha=0.7)
            self.axes[1].plot(self.df['dates'], self.df["dea"], color='b', alpha=0.7)
            self.axes[1].plot(self.df['dates'], self.df["dif"], color='black', alpha=0.7)
            self.axes[1].set_ylabel('macd', color='w')

    def _plot_custom_subplot2(self):
        self.axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(self.format_date))
        if self.theme:
            for index, i in enumerate(self.list):
                self.axes[2].plot(self.df["dates"], (index + 1) * self.df[i], color=self.colorPool[index], alpha=0.7)
        else:
            for index, i in enumerate(self.list):
                self.axes[2].bar(self.df["dates"], (index + 1) * self.df[i], color=self.colorPool[index], alpha=0.7)

    def show_chart(self):
        if self.df is None:
            self.df = self._read_clean_data()
            self._calculate_ma()

        if self.figure is None:
            self.figure = plt.figure(figsize=(12, 9), facecolor='#ffffff')
            self.axes.append(plt.subplot(4, 1, (1, 2)))
            self.axes[0].patch.set_facecolor('#ffffff')
            self.axes.append(plt.subplot(4, 1, 3))
            self.axes[1].patch.set_facecolor('#ffffff')
            self.axes.append(plt.subplot(4, 1, 4))
            self.axes[2].patch.set_facecolor('#ffffff')

        self._plot_candlestick()
        self._plot_custom_subplot1()
        self._plot_custom_subplot2()

        plt.show()


class StockChart3(StockChart):
    def __init__(self, path, name1, list, themeplot=False, draw_number=-500):
        super().__init__(path, name1, list, themeplot, draw_number)
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ma_data = {}

    def _calculate_ma(self):
        self.ma_data['5'] = self.df.Close.rolling(5).mean()
        self.ma_data['8'] = self.df.Close.rolling(20).mean()
        self.ma_data['13'] = self.df.Close.rolling(30).mean()
        self.ma_data['21'] = self.df.Close.rolling(60).mean()
        self.ma_data['55'] = self.df.Close.rolling(120).mean()
        self.ma_data['144'] = self.df.Close.rolling(250).mean()

    def _plot_candlestick(self):
        super()._plot_candlestick()

        for ma in ['5', '8', '13', '21', '55', '144']:
            self.axes[0].plot(self.df['dates'], self.ma_data[ma], label=ma)

    def _plot_custom_subplot1(self):
        self.axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(self.format_date))
        if self.name1 == "williams_r":
            self.axes[1].plot(self.df['dates'], self.df["williams_r"], color='b', alpha=0.7)
            self.axes[1].set_ylabel('williams', color='w')
        elif self.name1 == 'CCI':
            self.axes[1].plot(self.df['dates'], self.df["CCI"], color='b', alpha=0.7)
            self.axes[1].plot(self.df['dates'], self.df["CCI_HHV"], color='r', alpha=0.7)
            self.axes[1].set_ylabel('CCI', color='w')
        else:
            self.df['up'] = self.df.apply(lambda row: 1 if row['macd'] > 0 else 0, axis=1)
            self.axes[1].bar(self.df.query('up == 1')['dates'], self.df.query('up == 1')['macd'], color='r', alpha=0.7)
            self.axes[1].bar(self.df.query('up == 0')['dates'], self.df.query('up == 0')['macd'], color='g', alpha=0.7)
            self.axes[1].plot(self.df['dates'], self.df["dea"], color='b', alpha=0.7)
            self.axes[1].plot(self.df['dates'], self.df["dif"], color='black', alpha=0.7)
            self.axes[1].set_ylabel('macd', color='w')

    def _plot_custom_subplot2(self):
        self.axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(self.format_date))
        if self.theme:
            for index, i in enumerate(self.list):
                self.axes[2].plot(self.df["dates"], (index + 1) * self.df[i], color=self.colorPool[index], alpha=0.7)
        else:
            for index, i in enumerate(self.list):
                self.axes[2].bar(self.df["dates"], (index + 1) * self.df[i], color=self.colorPool[index], alpha=0.7)

    def show_chart(self):
        if self.df is None:
            self.df = self._read_clean_data()
            self._calculate_ma()

        if self.figure is None:
            self.figure = plt.figure(figsize=(12, 9), facecolor='#ffffff')
            self.axes.append(plt.subplot(4, 1, (1, 2)))
            self.axes[0].patch.set_facecolor('#ffffff')
            self.axes.append(plt.subplot(4, 1, 3))
            self.axes[1].patch.set_facecolor('#ffffff')
            self.axes.append(plt.subplot(4, 1, 4))
            self.axes[2].patch.set_facecolor('#ffffff')

        self._plot_candlestick()
        self._plot_custom_subplot1()
        self._plot_custom_subplot2()

        # Convert the plot to HTML format
        html_fig = mpld3.fig_to_html(self.figure)

        return html_fig

    def save_chart_to_html(chart, filename):
        html_fig = chart.show_chart()
        with open(filename, 'w') as file:
            file.write(html_fig)


class StockChart4(StockChart):
    def __init__(self, path, name1, list, themeplot=False, draw_number=-500):
        super().__init__(path, name1, list, themeplot, draw_number)
        self.ma_data = {}

    def _calculate_ma(self):
        self.ma_data['5'] = self.df.Close.rolling(5).mean()
        self.ma_data['8'] = self.df.Close.rolling(20).mean()
        self.ma_data['13'] = self.df.Close.rolling(30).mean()
        self.ma_data['21'] = self.df.Close.rolling(60).mean()
        self.ma_data['55'] = self.df.Close.rolling(120).mean()
        self.ma_data['144'] = self.df.Close.rolling(250).mean()

    def _plot_candlestick(self):
        super()._plot_candlestick()

        for ma in ['5', '8', '13', '21', '55', '144']:
            self.figure.add_trace(go.Scatter(x=self.df['dates'], y=self.ma_data[ma], name=ma))

    def _plot_custom_subplot1(self):
        if self.name1 == "williams_r":
            self.figure.add_trace(go.Scatter(x=self.df['dates'], y=self.df["williams_r"], name='williams_r'))
        elif self.name1 == 'CCI':
            self.figure.add_trace(go.Scatter(x=self.df['dates'], y=self.df["CCI"], name='CCI'))
            self.figure.add_trace(go.Scatter(x=self.df['dates'], y=self.df["CCI_HHV"], name='CCI_HHV'))
        else:
            self.df['up'] = self.df.apply(lambda row: 1 if row['macd'] > 0 else 0, axis=1)
            self.figure.add_trace(
                go.Bar(x=self.df.query('up == 1')['dates'], y=self.df.query('up == 1')['macd'], name='up',
                       marker_color='red'))
            self.figure.add_trace(
                go.Bar(x=self.df.query('up == 0')['dates'], y=self.df.query('up == 0')['macd'], name='down',
                       marker_color='green'))
            self.figure.add_trace(go.Scatter(x=self.df['dates'], y=self.df["dea"], name='dea'))
            self.figure.add_trace(go.Scatter(x=self.df['dates'], y=self.df["dif"], name='dif'))

    def _plot_custom_subplot2(self):
        if self.theme:
            for index, i in enumerate(self.list):
                self.figure.add_trace(go.Scatter(x=self.df["dates"], y=(index + 1) * self.df[i], name=i))
        else:
            for index, i in enumerate(self.list):
                self.figure.add_trace(go.Bar(x=self.df["dates"], y=(index + 1) * self.df[i], name=i))

    def show_chart(self):
        if self.df is None:
            self.df = self._read_clean_data()
            self._calculate_ma()

        self.figure = go.Figure()

        self._plot_candlestick()
        self._plot_custom_subplot1()
        self._plot_custom_subplot2()

        # Convert the plot to HTML format
        html_fig = self.figure.to_html()

        return html_fig


class StockChart5(StockChart):
    def __init__(self, path, name1, name2, draw_number=-500):
        super().__init__(path, None, None, False, draw_number)
        self.name1 = name1
        self.name2 = name2

    def _plot_bivariate_distribution(self):
        fig = px.density_contour(
            self.df,
            x=self.name1,
            y=self.name2,
            title=f"Bivariate Distribution of {self.name1} and {self.name2}",
            labels={self.name1: self.name1, self.name2: self.name2}
        )

        return fig

    def show_chart(self):
        if self.df is None:
            self.df = self._read_clean_data()

        self.figure = self._plot_bivariate_distribution()

        # Create the 'Html' directory if it doesn't exist
        if not os.path.exists('Html'):
            os.makedirs('Html')

        # Save the plot as an HTML file
        html_file = os.path.join('Html', 'bivariate_distribution.html')
        self.figure.write_html(html_file)

        return html_file


"""散点图"""


class StockChart6(StockChart):
    def __init__(self, path, name1, name2, draw_number=-500):
        super().__init__(path, None, None, False, draw_number)
        self.name1 = name1
        self.name2 = name2

    def _plot_bivariate_distribution(self):
        fig = go.Figure(data=go.Scatter(
            x=self.df[self.name1],
            y=self.df[self.name2],
            mode='markers'
        ))

        fig.update_layout(
            title=f"Bivariate Distribution of {self.name1} and {self.name2}",
            xaxis_title=self.name1,
            yaxis_title=self.name2
        )

        return fig

    def show_chart(self):
        if self.df is None:
            self.df = self._read_clean_data()

        self.figure = self._plot_bivariate_distribution()

        # Create the 'Html' directory if it doesn't exist
        if not os.path.exists('Html'):
            os.makedirs('Html')

        # Save the plot as an HTML file
        html_file = os.path.join('Html', 'bivariate_distribution.html')
        self.figure.write_html(html_file)

        return html_file
