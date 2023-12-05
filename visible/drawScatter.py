import pandas as pd
import os
import plotly.graph_objects as go
import webbrowser

"""散点图"""


class StockChart6(object):
    def __init__(self, path, name1, name2, draw_number=-500):
        self.path = path
        self.draw_number = draw_number
        self.name1 = name1
        self.name2 = name2
        self.df = self._read_clean_data()
        self.figure = self._plot_bivariate_distribution()

    def _read_clean_data(self):
        path = self.path
        df = pd.read_csv(path)
        if self.draw_number < 0:
            df = df.iloc[self.draw_number:]
        else:
            df = df.iloc[:self.draw_number]
        df = df.sort_values(by='Open time', ascending=True)
        return df

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

        if not os.path.exists('Html'):
            os.makedirs('Html')

        html_file = os.path.join('Html', 'bivariate_distribution.html')
        self.figure.write_html(html_file)

        webbrowser.open(html_file)

        return html_file
