import sys
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from io import BytesIO


def calculate_rate(df, name, returns_columns=None):
    if returns_columns is None:
        returns_columns = ['Return_1', 'Return_2', 'Return_3', 'Return_5']
    winning_probability = []

    for col in returns_columns:
        winning_probability.append(df[df[name] > 0][col].gt(0).mean())

    day_returns1 = df[df[name] > 0]['Return_1'].mean()
    days_returns2 = df[df[name] > 0]['Return_2'].mean()
    print(winning_probability)
    return winning_probability


class StockChartWidget(QWidget):
    def __init__(self, df, name1, name2, draw_number=-500):
        super().__init__()

        self.draw_number = draw_number
        self.name1 = name1
        self.name2 = name2
        self.df = df
        self._read_clean_data()

        self.layout = QVBoxLayout(self)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.layout.addWidget(self.view)

        self.setStyleSheet("background-color: #1e1f22;")  # 设置窗口背景色

        self._plot_bivariate_distribution()

    def _read_clean_data(self):
        if self.draw_number < 0:
            self.df = self.df.iloc[self.draw_number:]
        else:
            self.df = self.df.iloc[:self.draw_number]
        self.df = self.df.sort_values(by='Open time', ascending=True)

    def _plot_bivariate_distribution(self):
        # Scatter Plot
        scatter_fig, ax = plt.subplots()
        ax.scatter(self.df[self.name1], self.df[self.name2], c='y')
        ax.set_title(f"Scatter Plot of {self.name1} and {self.name2}")
        self._set_plot_background_color(ax)  # 设置图形背景色
        scatter_img = self._save_plot_to_image(scatter_fig)
        scatter_item = QGraphicsPixmapItem(QPixmap.fromImage(scatter_img))
        self.scene.addItem(scatter_item)

        # Bivariate Kernel Density Plot
        kde_fig, ax = plt.subplots()
        sns.kdeplot(data=self.df, x=self.name1, y=self.name2, fill=True, cmap="summer", ax=ax)
        ax.set_title(f"Bivariate Kernel Density Plot of {self.name1} and {self.name2}")
        self._set_plot_background_color(ax)  # 设置图形背景色
        kde_img = self._save_plot_to_image(kde_fig)
        kde_item = QGraphicsPixmapItem(QPixmap.fromImage(kde_img))
        kde_item.setPos(scatter_img.width(), 0)
        self.scene.addItem(kde_item)

    def _set_plot_background_color(self, ax):
        ax.set_facecolor('#1e1f22')  # 设置图形背景色
        ax.tick_params(axis='both', colors='white')  # 设置坐标轴颜色
        ax.title.set_color('white')  # 设置标题颜色
        ax.xaxis.label.set_color('white')  # 设置x轴标签颜色
        ax.yaxis.label.set_color('white')  # 设置y轴标签颜色

    def _save_plot_to_image(self, fig):
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='#1e1f22')  # 设置图形背景色
        plt.close(fig)

        img_buf.seek(0)
        img = QImage()
        img.loadFromData(img_buf.read())
        return img


def show_feature_img(df, name1, name2, draw_number=200):
    """
    Args: 展示特征的分布相关性以及收益率
        df: 传入的dataframe
        draw_number: 画散点数量
    """
    calculate_rate(df, name1)
    app = QApplication(sys.argv)
    window = StockChartWidget(df, name1, name2, draw_number=draw_number)
    window.show()
    sys.exit(app.exec_())
