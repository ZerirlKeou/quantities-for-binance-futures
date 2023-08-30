import os
import pandas as pd
import joblib


class PredictionWrite:
    def __init__(self, order, model_filetype, type_name):
        self.model = self.load_model(order, model_filetype, type_name)
        self.df = pd.read_csv('test.csv')
        self.pre_df = self.connect_data()

        features = self.pre_df.drop(["Open time", "Return_1", "Return_2"], axis=1)
        predictions = self.predict_model(features)
        self.pre_df['prediction'] = predictions

    def load_model(self, order, model_filetype, type_name):
        base_path = 'model'
        model_path = os.path.join(base_path, model_filetype, f'{type_name}_model{order}.pkl')
        return joblib.load(model_path)

    def connect_data(self):
        split_index = int(len(self.df) * 0.65)
        prediction_df = self.df[split_index:].copy()
        return prediction_df

    def predict_model(self, data):
        return self.model.predict(data)

    def write_csv(self):
        self.pre_df.to_csv("test.csv", index=False)
