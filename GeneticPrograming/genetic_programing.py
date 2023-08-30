import pandas as pd
import math
import csv
from GeneticPrograming import function_set


class GeneticProgramming(function_set.FunctionSet):
    def __init__(self):
        super().__init__()
        self.population_size = 50
        self.num_generations = 100

    def load_data(self):
        df = pd.read_csv('data\\5m_data.csv')
        return df

    # 定义适应度函数
    def fitness(self, alpha1, alpha2):
        a = pd.Series(alpha1)
        b = pd.Series(alpha2)
        # 计算与涨跌值相关的相关系数
        corralation = a.corr(b)
        return corralation

    def sorted_write(self, new_list):
        values = [row[0] for row in new_list]
        has_nan = any(math.isnan(value) for value in values)

        if has_nan:
            new_list = [row for row in new_list if not math.isnan(row[0])]
        sorted_data = sorted(new_list, key=lambda x: x[0])
        csv_file = "sorted_data.csv"

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(sorted_data)

        print("CSV文件写入完成。")

    def eliminate_half(self, lst):
        lst.sort(reverse=True, key=lambda x: x[0])
        half_idx = len(lst) // 2
        lst = lst[:half_idx]
        return lst

    def evolution(self, times, population, df):
        score = []
        for k in range(len(population)):
            p = k + 1
            while p < len(population):
                alpha1 = population[k]
                alpha2 = population[p]
                for j in self.function_Two:
                    alpha_j = j(alpha1, alpha2)
                    score.append([self.fitness(alpha_j, df['Signal']), j, k, p])
                p = p + 1
        return score

    def gep_train(self):
        df = self.load_data()
        population = [df['Close'], df['Open'], df['High'], df['Low'], df['Volume'], df['Quote asset volume'],
                      df['Number of trades'], df['Taker buy base asset volume'], df['Taker buy quote asset volume'],
                      df['Ignore'], df['dif'], df['dea'], df['macd'], df['williams_r'], df['CCI'], df['volume_change']]
        # 种群迭代
        score = self.evolution(times=1, population=population, df=df)
        # 末位淘汰
        new_list = self.eliminate_half(score)
        # 排序与写入
        self.sorted_write(new_list)

    # print(fitness(list(map(sigmoid,new_list[0][1](population[new_list[0][2]],population[new_list[0][3]]))),df['Signal']))
