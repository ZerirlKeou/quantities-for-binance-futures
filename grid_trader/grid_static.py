class InitGrid(object):
    """启动网格策略类，返回买入栈和卖出栈
    : params close : 传入目前收盘价作为参考建立网格
    : params low_price : 网格最低价格
    : params high_price : 网格最高价格
    : params size : 划分为size个网格，决定网格区间大小
    """
    def __init__(self, close, low_price, high_price, size):
        self.grid = []
        self.size = size
        self.interval_size = (high_price - low_price) / size
        self._pair_stack(close, low_price, high_price, self.interval_size)

    def _pair_stack(self, close, low_price, high_price, interval_size):
        for i in range(1, self.size + 1):
            self.grid.append(low_price + i * interval_size)
        for i in range(self.size):
            if low_price < close < high_price:
                if self.grid[i] > close:
                    self.buy_stack = self.grid[:i]
                    self.sell_stack = self.grid[i:]
                    break
            else:
                raise ValueError("输入目前价格不在网格区间内，请检查输入!")


def test_class():
    grid = InitGrid(close=241.5, low_price=232.45, high_price=257.56, size=21)
    print(grid.buy_stack[-1:])
    original_list = [2, 4, 5, 6]
    number_to_compare = 3

    result = [num for num in original_list if num > number_to_compare]
    print(result)
