import pandas as pd
import numpy as np
from models.predictive_model import PredictiveModel

from test import knn  # here import the model to be tunned

params = {'use_recency': True, 'n_most_recent': 2000, 'classifier': knn, 'update_frequency': 1, 'debug': True,
          'use_calibration': True}  # params to initialize PredictiveModel class

features_ = pd.read_csv('../features.csv')
dataset = pd.read_csv('../training_data.csv', parse_dates=['Date', 'Open'])


class Src(object):
    """
    Simulates class Data. It has same interface to communicate with PredictiveModel class.
    """

    def __init__(self, features, matches, num_tested_matches=1000):
        """
        :param features: pd.DataFrame:
            Contains generated features.
        :param matches: pd.DataFrame:
            Contains all matches. It supplies true labels
        :param num_tested_matches: int:
            Specifies how many matches should be used as test dataset.
        """
        self.features = features
        self.matches = matches
        self.tested_size = num_tested_matches
        self.opps_matches = [params['n_most_recent']]

    def return_values(self):
        """
        Returns actual 'opps' to be predicted. It is meant to be test dataset
        """
        return self.features[self.opps_matches[0]: self.opps_matches[0] + self.tested_size]


if __name__ == '__main__':

    src = Src(features_, dataset, num_tested_matches=1000)
    predictor = PredictiveModel(src, **params)
    step = 250  # meant as matches

    for i in range(params['n_most_recent'] + step, features_.shape[0] - 2*src.tested_size, step):
        print(f'Training on matches from {i - step - params["n_most_recent"]} to {i - step - 1} \n'
              f'Testing on matches from {i - step} to {i - step + src.tested_size - 1}')
        predictor.run_iter([], src.return_values())
        predictor.test_me(src.return_values(), src.matches.loc[src.opps_matches[0]: src.opps_matches[0] + src.tested_size - 1,
                                               ['H', 'D', 'A']])
        src.opps_matches = [i]
        print('################################################### \n')
