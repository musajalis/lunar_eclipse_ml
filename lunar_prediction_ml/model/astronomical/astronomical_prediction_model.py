from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from pathlib import Path
import json
import numpy as np
import scipy.optimize
from datetime import date


class DataArrangeTool:
    def __init__(self, astronomical: str) -> None:
        self.astronomical = astronomical
        self.time = []
        self.dist_x = []
        self.dist_y = []
        self.moon_phase = []
        self.arc_min = []

    @staticmethod
    def load_data() -> dict:
        data_path = os.path.join(Path(os.getcwd()).parent, 'data')
        try:
            with open(os.path.join(data_path, 'processed_data.json'), 'r') as file:
                data = json.load(file)
        except IOError:
            raise Exception("I/O error")
        return data

    def set_parameters(self) -> None:
        data = self.load_data()
        astronomical_data = data['astronomical'][self.astronomical]
        self.dist_x = astronomical_data['distance'] * np.cos(astronomical_data['ecliptic_lon'])
        self.dist_y = astronomical_data['distance'] * np.sin(astronomical_data['ecliptic_lon'])
        self.time = data['time']['ord']

        if self.astronomical == 'moon':
            self.moon_phase = astronomical_data['phase_portion']
            self.arc_min = astronomical_data['arc_min']
        elif self.astronomical == 'sun':
            self.arc_min = astronomical_data['arc_min']

    @staticmethod
    def save_data(data: np.array, evaluation: bool = False) -> None:
        if evaluation:
            data_path = os.path.join(Path(os.getcwd()).parent, 'evaluation')
            file = 'evaluation.json'
        else:
            data_path = os.path.join(Path(os.getcwd()).parent, 'data')
            file = 'predicted_data.json'
        try:
            with open(os.path.join(data_path, file), 'w') as fw:
                json.dump(data, fw, indent=4)
        except IOError:
            raise Exception("I/O error")


class AstronomicalPredictionModel(DataArrangeTool):

    def __init__(self, astronomical: str) -> None:
        super().__init__(astronomical)
        self.covariance = []
        self.amplitude = 0
        self.omega = 0
        self.phi = 0
        self.offset = 0
        self.frequency = 0
        self.period = 0
        self.x_ = []
        self.x_t = []
        self.y_ = []
        self.y_t = []
        self.y_pred = []

    def split_test_train(self, test_size=0.2, axis='x'):
        if axis == 'x':
            dist = self.dist_x
        elif axis == 'y':
            dist = self.dist_y
        else:
            raise Exception("Write x or y only")

        time_, dist_ = np.array(self.time), np.array(dist)

        self.x_, self.x_t, self.y_, self.y_t = train_test_split(time_,
                                                                dist_,
                                                                test_size=test_size,
                                                                shuffle=False)

    def generate_sampling(self,
                          n: int = 200,
                          step: float = 0.1,
                          sigma: float = 1.0,
                          add_noise: bool = True) -> (np.array, np.array, np.array):
        sample_points = np.linspace(0, len(self.x_) - 1, n // 2, dtype='int', endpoint=False)
        sample_points_test = np.linspace(0, len(self.x_t) - 1, n // 2, dtype='int', endpoint=False)

        self.x_, self.y_ = self.x_[sample_points], self.y_[sample_points]
        self.x_t, self.y_t = self.x_t[sample_points_test], self.y_t[sample_points_test]

        if add_noise:
            self.x_t = self.x_t + sigma * np.random.random(len(self.x_t))

    @staticmethod
    def sin_function(t: int, a: float, w: float, phi: float, c: float) -> float:
        return a * np.sin(w * t + phi) + c

    def optimize_fourier(self) -> None:
        fft_x = np.fft.fftfreq(len(self.x_), d=(self.x_[1] - self.x_[0]))
        fft_y = abs(np.fft.fft(self.y_))
        freq_ = abs(fft_x[np.argmax(fft_y[1:]) + 1])
        amp_ = np.std(self.y_) * 2. ** 0.5
        offset_ = np.mean(self.y_)
        initial = np.array([amp_, 2 * np.pi * freq_, 0, offset_])
        params, self.covariance = scipy.optimize.curve_fit(
            self.sin_function,
            self.x_,
            self.y_,
            p0=initial,
            method='lm')

        self.amplitude, self.omega, self.phi, self.offset = params
        self.frequency = self.omega / (2 * np.pi)
        self.period = 1 / self.frequency

    def predict_position(self) -> None:
        assert (self.amplitude or self.omega or self.phi or self.offset), "Optimize First"

        self.y_pred = np.array(
            [self.sin_function(t, self.amplitude, self.omega, self.phi, self.offset) for t in self.x_t])

    def fetch_prediction(self, predict=True):
        if predict:
            return np.vstack((self.y_t, self.y_pred)).T.tolist()
        else:
            int_time = self.x_t.astype(np.int)
            return [date.fromordinal(i).strftime("%Y-%m-%d") for i in int_time]

    def evaluate(self) -> dict:
        evaluation = {
            "Optimal amplitude": self.amplitude,
            "Optimal omega": self.omega,
            "Optimal phi": self.phi,
            "Optimal offset": self.offset,
            "Optimal frequency": self.frequency,
            "Optimal period": self.period,
            "Covariance": self.covariance.tolist(),
            "Mean Absolute Error": metrics.mean_absolute_error(self.y_t, self.y_pred),
            "Mean Squared Error": metrics.mean_squared_error(self.y_t, self.y_pred),
            "Explained Variance Score": metrics.explained_variance_score(self.y_t, self.y_pred),
            "Median Absolute Error": metrics.median_absolute_error(self.y_t, self.y_pred),
            "R2 score": metrics.r2_score(self.y_t, self.y_pred),
            "Max Error": metrics.max_error(self.y_t, self.y_pred)}

        return evaluation

astronomical_bodies = ['sun', 'moon', 'mercury', 'mars', 'venus', 'jupiter', 'saturn']


def predict(astronomical:list = None, n=300):
    if astronomical is None:
        astronomical = astronomical_bodies

    pred_data = []
    eval_data = []
    for a in astronomical:
        pred_curr = {'time': [],
                      a: {'[true_dist, pred_dist]': {}}}
        eval_curr = {a: {}}
        pred = AstronomicalPredictionModel(a)
        pred.set_parameters()
        for axis in 'xy':
            pred.split_test_train(axis=axis)
            pred.generate_sampling(n)
            pred.optimize_fourier()
            pred.predict_position()

            pred_curr[a]['[true_dist, pred_dist]'][f'{axis}_dist'] = pred.fetch_prediction()
            pred_curr[a][f'{axis}_axis'] = pred.fetch_prediction(predict=True)
            eval_curr[a][f'{axis}_axis'] = pred.evaluate()
        pred_curr[a]['time'] = pred.fetch_prediction(predict=False)

        pred_data.append(pred_curr)
        eval_data.append(eval_curr)

    DataArrangeTool.save_data(pred_data, evaluation=False)
    DataArrangeTool.save_data(eval_data, evaluation=True)


