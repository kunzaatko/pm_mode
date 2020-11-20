from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

dataset = pd.read_csv('../training_data.csv', parse_dates=['Date', 'Open'])
seasons = [dataset[dataset['Sea'] == sea] for sea in dataset['Sea'].unique()]

train = pd.concat([seasons[i] for i in range(7)])
test = pd.concat([seasons[i] for i in range(7, len(seasons))])


class RandomForest(object):
    def __init__(self, data, update_frequency=1):
        # TODO we will need to normalize cols of dataframe in class Data before training
        self.data = data  # instance of class Data(), containing table which will be used as data to train
        self.rf_model = None
        self.update_frequency = update_frequency
        self.last_update = 0
        self.P_dis = None
        self.accuracy = None

    def _predict(self, opps):
        # we need to transform dataframe opps to structure as variable 'features' below in method _update_model here
        # or already in class Data
        to_be_predicted = transform(opps)
        self.P_dis = self.rf_model.predict_proba(to_be_predicted)

    def _update_model(self):
        # It is assumed that in class Data() exist dataframe containing final data
        # HID, AID needs to be one-hot encoded or not used
        features = self.data.final_data[['HID', 'AID', 'H_diff_matches_1', '... some additional features']].to_numpy()
        labels = self.data.final_data[['H', 'D', 'A']].to_numpy()

        rf = RandomForestClassifier(n_estimators=100)
        self.rf_model = rf.fit(features, np.argmax(labels, axis=1))  # H has index 0, D has index 1, away has index 2

        # for debugging and selecting important features
        feature_importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                     axis=0)
        indices = np.argsort(feature_importances)[::-1]
        # Print the feature ranking
        print("Feature ranking based on feature_importances (MDI):")
        for f in range(features.shape[1]):
            print(" %d. feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))


    def run_iter(self, inc, opps):
            """
            Runs the iteration of the evaluation loop.
            :param inc: pd.DataFrame:
                Has to include 'HID', 'AID', 'HSC', 'ASC' columns.
            :param opps: pd.DataFrame:
                Has to contain 'HID', 'AID' column.
            :return: pd.DataFrame:
                DataFrame containing accuracies for all predictions and overall accuracy
            """
            if self.last_update % self.update_frequency == 0:
                self._update_model()
            self.last_update += 1
            self.P_dis = self._predict(opps)
            return self.accuracy
