import finplot as fplt
import pandas as pd
import numpy as np



def fig_config():
    fplt.legend_text_color = '#fff'
    fplt.background = '#1e1f22'
    fplt.foreground = '#fff'
    fplt.odd_plot_background = '#1b1c1f'
    fplt.candle_bear_color = fplt.candle_bear_body_color = '#2ebc9c'
    fplt.candle_bull_color = fplt.candle_bull_body_color = '#ef5350'
    fplt.volume_bull_color = fplt.volume_bull_body_color = '#f7a9a7'
    fplt.volume_bear_color = '#92d2cc'
    fplt.volume_neutral_color = '#bbb'
    fplt.cross_hair_color = '#fff'
    fplt.draw_line_color = '#fff'


def fig_show():
    fplt.autoviewrestore()
    fplt.show()


class DefTypesPool:
    def __init__(self):
        self.routes = {}

    def route_types(self, types_str):
        def decorator(f):
            self.routes[types_str] = f
            return f

        return decorator

    def route_output(self, path):
        function_val = self.routes.get(path)
        if function_val:
            return function_val
        else:
            raise ValueError('Route"{}""has not been registered'.format(path))


class MplTypesDraw:
    mpl = DefTypesPool()

    @mpl.route_types(u"kline")
    def kline_plot(self, df_dat, ax):
        fplt.candlestick_ochl(df_dat[['Open', 'Close', 'High', 'Low']], ax=ax)
        df_dat.Close.ewm(span=5).mean().plot(ax=ax, legend='EMA')
        df_dat.Close.ewm(span=10).mean().plot(ax=ax, legend='EMA')
        df_dat.Close.ewm(span=20).mean().plot(ax=ax, legend='EMA')

    @mpl.route_types(u"line")
    def line_plot(self, df_dat, ax):
        fplt.plot(df_dat, ax=ax)

    @mpl.route_types(u"bar")
    def bar_plot(self, df_bat, ax):
        fplt.volume_ocv(df_bat,ax,colorfunc=fplt.strength_colorfilter)

    @mpl.route_types(u"volume")
    def volume_plot(self, df_dat, ax):
        df_dat[['Open', 'Close', 'Volume']].plot(kind='volume', ax=ax)

    @mpl.route_types(u"macd")
    def macd_plot(self, df_dat, ax):
        df_dat['macd_1'] = df_dat['macd'].shift(1)
        df_dat['macd_copy'] = df_dat['macd']
        # print(df_dat[['macd_copy', 'macd_1', 'macd']])
        fplt.volume_ocv(df_dat[['macd_copy', 'macd_1', 'macd']], ax=ax, colorfunc=fplt.strength_colorfilter)
        fplt.plot(df_dat['dif'], ax=ax, legend='dif')
        fplt.plot(df_dat['dea'], ax=ax, legend='dea')

    @mpl.route_types(u"williams")
    def williams_plot(self, df_dat, ax):
        fplt.plot(df_dat['williams_r'], ax=ax, legend='williams')

    @mpl.route_types(u"cci")
    def cci_plot(self, df_dat, ax):
        fplt.plot(df_dat['CCI'], ax=ax, legend='CCI')
        fplt.plot(df_dat['ding'], ax=ax, legend='CCI_HHV')

    @mpl.route_types(u"zjfz")
    def zjfz_plot(self, df_dat, ax):
        fplt.plot(df_dat['VAR8'], ax=ax, legend='zjfz')

    @mpl.route_types(u"basic_Open_point")
    def basic_open_point(self, df_dat, ax):
        fplt.plot(df_dat['basic_Open_point'], ax=ax, legend='Basic Open Point')

    @mpl.route_types(u"macd_back")
    def macd_back(self, df_dat, ax):
        fplt.plot(df_dat['macd_back'], ax=ax, legend='MACD Back')

    @mpl.route_types(u"macd_sell")
    def macd_sell(self, df_dat, ax):
        fplt.plot(df_dat['macd_sell'], ax=ax, legend='MACD Sell')

    @mpl.route_types(u"golden")
    def golden_point(self, df_dat, ax):
        fplt.plot(df_dat['buypill'], ax=ax, legend='buypill')
        fplt.plot(df_dat['sellpill'], ax=ax, legend='sellpill')

    @mpl.route_types(u"stoch_rsi")
    def stoch_rsi_plot(self, df_dat, ax):
        fplt.plot(df_dat['KR'], ax=ax, legend='KR')
        fplt.plot(df_dat['DR'], ax=ax, legend='DR')
        # fplt.plot(df_dat['rsiBuy'], ax=ax, legend='rsiBuy')

    @mpl.route_types(u"fillTrade")
    def fillTrade_plot(self, df_dat, ax):
        skip_days = False
        df_dat['strategy'] = df_dat['basic_Open_point']
        for kl_index, value in df_dat['strategy'].iteritems():
            if (value == 1) and (skip_days == False):
                start = df_dat.index.get_loc(kl_index)
                skip_days = True
            elif (value == -1) and (skip_days == True):
                end = df_dat.index.get_loc(kl_index) + 1
                skip_days = False
                if df_dat['Close'][end-1] <= df_dat['Close'][start]:
                    print('end{}'.format(df_dat['Close'][end-1]))
                    print('start{}'.format(df_dat['Close'][start]))
                    pass
                    # fplt.plot(df_dat['Close'][start:end],ax=ax,color='g')
                else:
                    fplt.plot(df_dat['Close'][start:end],ax=ax,color='r')
    @mpl.route_types(u"money")
    def money_trade_back(self,df_dat, ax):
        fplt.plot(df_dat['money'], ax=ax, legend='Money')

class MplVisualIf(MplTypesDraw):

    def __init__(self):
        MplTypesDraw.__init__(self)
        self.df = None
        self.draw_kind = None
        self.axes = None
        self.title = None
        self.ax4 = None
        self.ax3 = None
        self.ax2 = None
        self.ax = None
        fig_config()

    def fig_create(self, **kwargs):
        if 'draw_kind' in kwargs.keys():
            if len(self.draw_kind) == 1:
                self.ax = fplt.create_plot(self.title, rows=1)
            elif len(self.draw_kind) == 2:
                self.ax, self.ax2 = fplt.create_plot(self.title, rows=2)
            elif len(self.draw_kind) == 3:
                self.ax, self.ax2, self.ax3 = fplt.create_plot(self.title, rows=3)
            elif len(self.draw_kind) == 4:
                self.ax, self.ax2, self.ax3, self.ax4 = fplt.create_plot(self.title, rows=4)
            else:
                print(u"Chose one draw kind firstly")
        else:
            self.ax, self.ax2 = fplt.create_plot(self.title, rows=2)

    def fig_output(self, **kwargs):
        if 'path' in kwargs.keys():
            self.df = pd.read_csv(kwargs.get('path'), parse_dates=True, index_col='Open time')
        elif 'df' in kwargs.keys():
            self.df = kwargs.get('df')
        else:
            print(u"There is no path for dataframe to draw")
        self.draw_kind = kwargs.get('draw_kind', [])
        self.title = kwargs.get('title', 'Default Title')
        self.fig_create(**kwargs)
        self.axes = [self.ax, self.ax2, self.ax3, self.ax4]
        for i, ax in enumerate(self.axes):
            if i < len(self.draw_kind):
                view_function = self.mpl.route_output(self.draw_kind[i])
                view_function(self, df_dat=self.df, ax=self.axes[i])
        fig_show()