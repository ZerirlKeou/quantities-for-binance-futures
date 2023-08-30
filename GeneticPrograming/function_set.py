import math

class FunctionSet:
    def __init__(self):
        self.function_Two = [self.multi, self.subtract, self.add]
        self.function_LastSet = [self.sigmoid, self.absolute, self.sine, self.cos, self.tan, self.sqrt, self.logarithm]
        self.golden_nums = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    def multi(self, a, b):
        return a * b

    def subtract(self, a, b):
        return a - b

    def add(self, a, b):
        return a + b

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def absolute(self, x):
        return abs(x)

    def sine(self, x):
        return math.sin(x)

    def cos(self, x):
        return math.cos(x)

    def tan(self, x):
        return math.tan(x)

    def sqrt(self, x):
        return math.sqrt(x)

    def exp(self, x):
        return math.exp(x)

    def logarithm(self, x):
        return math.log(x)