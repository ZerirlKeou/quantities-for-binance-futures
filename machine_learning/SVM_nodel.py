import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
# import cuml.svm


class deal_with_df:
    def __init__(self):
        pass

    # 分类器准备数据
    def connect_data1(self, threshold=0.0010):
        df = pd.read_csv('test.csv')
        features = df.drop(["Open time", "Return_1", "Return_2"], axis=1)
        targets = np.where(df['Return_1'] > threshold, 1, np.where(df['Return_1'] < -threshold, -1, 0))

        # 根据时间顺序划分训练集和测试集
        split_index = int(len(df) * 0.35)  # 根据前80%的索引位置进行划分
        X_train = features[:split_index]
        y_train = targets[:split_index]
        X_test = features[split_index:]
        y_test = targets[split_index:]

        return X_train, y_train, X_test, y_test

    # 回归器准备数据
    def connect_data2(self):
        df = pd.read_csv('test.csv')
        features = df.drop(["Open time", "Return_1", "Return_2"], axis=1)
        targets = df['Return_1']  # 连续的目标变量

        # 根据时间顺序划分训练集和测试集
        split_index = int(len(df) * 0.35)  # 根据前80%的索引位置进行划分
        X_train = features[:split_index]
        y_train = targets[:split_index]
        X_test = features[split_index:]
        y_test = targets[split_index:]

        return X_train, y_train, X_test, y_test

    def get_feature_names(self):
        df = pd.read_csv('test.csv')
        feature_names = df.drop(["Open time", "Return_1", "Return_2"], axis=1).columns.tolist()
        return feature_names


def train_support_vector_machine_classifier():
    dl = deal_with_df()
    X_train, y_train, X_test, y_test = dl.connect_data1()

    # 训练支持向量机模型
    svm_model = SVC(kernel='linear', C=1.0, gamma=0.1, random_state=0)
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    predictions = svm_model.predict(X_test)
    joblib.dump(svm_model, 'model/svm/svm_model1.pkl')

    # 输出分类报告
    report = classification_report(y_test, predictions)

    # 绘制特征重要性图（SVM不直接提供特征重要性，这里使用特征系数绝对值作为示例）
    feature_importance = abs(svm_model.coef_[0])

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    plt.xticks(range(len(feature_importance)), dl.get_feature_names(), rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_img\\feature_importance_svc.png')

    return report


def train_support_vector_machine_regression():
    dl = deal_with_df()
    X_train, y_train, X_test, y_test = dl.connect_data1()

    # 训练支持向量机回归模型
    svm_model = SVR(kernel='linear', C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    predictions = svm_model.predict(X_test)
    joblib.dump(svm_model, 'model/svm/svm_model2.pkl')

    # 输出回归报告
    report = "Regression Report:\n"
    report += f"R2 Score: {svm_model.score(X_test, y_test)}\n"

    # 绘制特征重要性图（SVM不直接提供特征重要性，这里不可用）
    report += "Feature Importance: N/A\n"

    return report


def train_svm_with_grid_search():
    dl = deal_with_df()
    X_train, y_train, X_test, y_test = dl.connect_data1()

    # 定义SVM模型
    svm_model = SVC()

    # 定义参数网格
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10]
    }

    # 使用网格搜索进行参数调优
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 输出最佳参数组合和对应的模型性能
    best_params = grid_search.best_params_
    print("Best Parameters: ", best_params)

    # 使用最佳参数重新训练模型
    svm_model = SVC(**best_params)
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    predictions = svm_model.predict(X_test)
    joblib.dump(svm_model, 'model/svm/svm_model.pkl')

    # 输出分类报告
    report = classification_report(y_test, predictions)
    return report
