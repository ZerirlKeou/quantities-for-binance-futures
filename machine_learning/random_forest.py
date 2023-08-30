import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed


class DealWithDF:
    def __init__(self):
        self.df = pd.read_csv('test.csv')

    def connect_data(self, threshold=0.0010, return_type="classifier"):
        """
        :params return_type: 返回的是分类器数据还是回归器数据,可选为"classifier","regressor"
        :params threshold: 输入的为第二天上涨的百分比,默认为0.8%
        """
        features = self.df.drop(["Open time", "Return_1", "Return_2"], axis=1)

        if return_type == "classifier":
            targets = np.where(self.df['Return_1'] > threshold, 1, np.where(self.df['Return_1'] < -threshold, -1, 0))
        elif return_type == "regressor":
            targets = self.df['Return_1']
        else:
            raise ValueError("Invalid return_type. Must be 'classifier' or 'regressor'.")

        split_index = int(len(self.df) * 0.8)
        X_train = features[:split_index]
        y_train = targets[:split_index]
        X_test = features[split_index:]
        y_test = targets[split_index:]

        return X_train, y_train, X_test, y_test

    def get_feature_names(self):
        feature_names = self.df.drop(["Open time", "Return_1", "Return_2"], axis=1).columns.tolist()
        return feature_names


def train_random_forest_classifier():
    dl = DealWithDF()
    X_train, y_train, X_test, y_test = dl.connect_data()

    # 训练随机森林模型
    # rf = RandomForestClassifier(max_depth=5, min_samples_split=34, n_estimators=2500,
    #                             random_state=0)
    rf = RandomForestClassifier(n_estimators=100,
                                random_state=0)
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    predictions = rf.predict(X_test)
    joblib.dump(rf, 'model/randomforest/random_forest_model4.pkl')

    # 输出分类报告
    report = classification_report(y_test, predictions)

    # 计算特征重要性
    feature_importance = rf.feature_importances_

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    plt.xticks(range(len(feature_importance)), dl.get_feature_names(), rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_img\\feature_importance.png')

    return report


def train_isolation_random_forest_regressor():
    dl = DealWithDF()
    X_train, y_train, X_test, y_test = dl.connect_data(return_type="regressor")

    # 训练孤立随机森林模型
    isof = IsolationForest(max_samples='auto', random_state=0)
    isof.fit(X_train)

    # 在训练集和测试集上进行预测
    train_predictions = isof.predict(X_train)
    test_predictions = isof.predict(X_test)

    # 保存模型
    joblib.dump(isof, 'model\\randomforest\\isolation_random_forest_model.pkl')

    # 输出评估指标
    outlier_scores = isof.decision_function(X_test)
    # 计算离群样本的百分比
    outlier_percentage = (outlier_scores < 0).mean()

    # 计算 ROC AUC
    roc_auc = roc_auc_score(y_test, -outlier_scores)  # 注意，取负号以使得离群得分越大越好
    print(outlier_scores)
    print(outlier_percentage)
    print(roc_auc)

    return outlier_scores, outlier_percentage, roc_auc


def train_random_forest_regressor():
    dl = DealWithDF()
    X_train, y_train, X_test, y_test = dl.connect_data(return_type="regressor")
    # max_depth=8, min_samples_split=10, n_estimators=2050,
    #                        random_state=00
    # 训练随机森林模型
    rf = RandomForestRegressor(max_depth=5, min_samples_split=34, n_estimators=2500,
                               random_state=0)
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    predictions = rf.predict(X_test)
    joblib.dump(rf, 'model\\randomforest\\random_forest_model3.pkl')

    # 输出回归模型的评估指标
    r2 = r2_score(y_test, predictions)

    return r2


def search_best_parameters_regressor():
    dl = DealWithDF()
    X_train, y_train, X_test, y_test = dl.connect_data(return_type="regressor")

    # 定义要搜索的超参数范围
    param_space = {
        'max_depth': (5, 40),
        'min_samples_split': (2, 40),
        'n_estimators': (100, 2500),
        'random_state': [0]
    }

    # 创建随机森林回归器
    rf = RandomForestRegressor()

    # 创建BayesSearchCV对象
    bayes_search = BayesSearchCV(estimator=rf, search_spaces=param_space, scoring='r2', cv=5)

    # 在训练集上进行超参数搜索
    bayes_search.fit(X_train, y_train)

    # 获取最佳模型和最佳参数组合
    best_rf = bayes_search.best_estimator_
    best_params = bayes_search.best_params_

    # 输出最佳参数组合
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # 在测试集上进行预测
    predictions = best_rf.predict(X_test)
    joblib.dump(best_rf, 'model\\randomforest\\random_forest_model3.pkl')

    # 输出回归模型的评估指标
    r2 = r2_score(y_test, predictions)

    return r2


def search_best_paramsbyes(rf_type_classifier=True):
    # 定义超参数空间
    param_space = {
        'n_estimators': Integer(100, 2500),
        'max_depth': Integer(5, 40),
        'min_samples_split': Integer(2, 40)
    }
    dl = DealWithDF()
    if rf_type_classifier:
        X_train, y_train, X_test, y_test = dl.connect_data()

        # 创建随机森林分类器
        rf = RandomForestClassifier(random_state=0)

    else:
        print('Begin regressor parameters searching work')
        X_train, y_train, X_test, y_test = dl.connect_data(return_type="regressor")

        # 创建随机森林分类器
        rf = RandomForestRegressor(random_state=0)

    # 创建贝叶斯搜索对象
    bayes_search = BayesSearchCV(estimator=rf, search_spaces=param_space, cv=3, scoring='accuracy', n_jobs=-1)

    # 在训练集上进行超参数搜索
    bayes_search.fit(X_train, y_train)

    # 输出最佳超参数组合和得分
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    # 使用最佳超参数组合训练最终模型
    if rf_type_classifier:
        best_rf = RandomForestClassifier(random_state=0, **best_params)
    else:
        best_rf = RandomForestRegressor(random_state=0, **best_params)
    best_rf.fit(X_train, y_train)

    return best_rf


def draw_heat_picture(rf_type_classifier=True):
    dl = DealWithDF()
    if rf_type_classifier:
        X_train, y_train, X_test, y_test = dl.connect_data()

    else:
        print('Begin classifier parameters searching work')
        X_train, y_train, X_test, y_test = dl.connect_data(return_type="classifier")

    # 定义不同超参数值
    n_estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500,
                    2750]
    max_features = ['auto']

    # 存储评估指标数据的列表
    accuracy_data = []
    precision_data = []
    recall_data = []
    f1_score_data = []

    # 定义并行化函数
    def train_model_and_evaluate(n_estimator, max_feature):
        # 创建随机森林分类器
        rf = RandomForestClassifier(n_estimators=n_estimator, random_state=0, max_features=max_feature)

        # 训练模型
        rf.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = rf.predict(X_test)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 返回评估指标数据
        return accuracy, precision, recall, f1

    # 使用并行化进行训练和评估
    results = Parallel(n_jobs=-1)(
        delayed(train_model_and_evaluate)(n_estimator, max_feature)
        for n_estimator in n_estimators
        for max_feature in max_features
    )

    # 提取评估指标数据
    accuracy_data, precision_data, recall_data, f1_score_data = zip(*results)

    # 将数据转换为NumPy数组
    accuracy_data = np.array(accuracy_data).reshape(len(n_estimators), len(max_features))
    precision_data = np.array(precision_data).reshape(len(n_estimators), len(max_features))
    recall_data = np.array(recall_data).reshape(len(n_estimators), len(max_features))
    f1_score_data = np.array(f1_score_data).reshape(len(n_estimators), len(max_features))

    # 创建图形
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    # 绘制Accuracy曲线
    axes[0].set_title('Accuracy')
    im1 = axes[0].imshow(accuracy_data, cmap='viridis')
    axes[0].set_xticks(np.arange(len(max_features)))
    axes[0].set_yticks(np.arange(len(n_estimators)))
    axes[0].set_xticklabels(max_features)
    axes[0].set_yticklabels(n_estimators)
    axes[0].set_xlabel('Max Features')
    axes[0].set_ylabel('N Estimators')
    plt.colorbar(im1, ax=axes[0])

    # 绘制Precision曲线
    axes[1].set_title('Precision')
    im2 = axes[1].imshow(precision_data, cmap='viridis')
    axes[1].set_xticks(np.arange(len(max_features)))
    axes[1].set_yticks(np.arange(len(n_estimators)))
    axes[1].set_xticklabels(max_features)
    axes[1].set_yticklabels(n_estimators)
    axes[1].set_xlabel('Max Features')
    axes[1].set_ylabel('N Estimators')
    plt.colorbar(im2, ax=axes[1])

    # 绘制Recall曲线
    axes[2].set_title('Recall')
    im3 = axes[2].imshow(recall_data, cmap='viridis')
    axes[2].set_xticks(np.arange(len(max_features)))
    axes[2].set_yticks(np.arange(len(n_estimators)))
    axes[2].set_xticklabels(max_features)
    axes[2].set_yticklabels(n_estimators)
    axes[2].set_xlabel('Max Features')
    axes[2].set_ylabel('N Estimators')
    plt.colorbar(im3, ax=axes[2])

    # 绘制F1 Score曲线
    axes[3].set_title('F1 Score')
    im4 = axes[3].imshow(f1_score_data, cmap='viridis')
    axes[3].set_xticks(np.arange(len(max_features)))
    axes[3].set_yticks(np.arange(len(n_estimators)))
    axes[3].set_xticklabels(max_features)
    axes[3].set_yticklabels(n_estimators)
    axes[3].set_xlabel('Max Features')
    axes[3].set_ylabel('N Estimators')
    plt.colorbar(im4, ax=axes[3])

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("feature_img\\test2.jpg", dpi=250, format="jpg")
    # 显示图形
    plt.show()
