import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


class PredictiveModel(object):
    def __init__(self, data, classifier='rf', update_frequency=1):
        """
        :param data: Data()
            Instnce of
        :param classifier: clf of model
            Specifies the classifier used to predictions:
                        'rf' - Random forest
                        'ab' - AdaBoost
                        'gb' - GradientBoost
                        'xgb' - XGBoost (Extreme gradient boosting)
                        'nb' - Gaussian NaiveBayes
                        'knn' - K-nearest-neighbours
                        'lr' - Logistic regression
        :param update_frequency: int
            Specifies how often to re-train model
        """
        # TODO we will need to normalize cols of dataframe in class Data before training
        self.data = data  # instance of class Data(), containing table which will be used as data to train
        self.clf = classifier  # specifies type of model
        self.predictive_model = None  # here is stored already fitted model
        self.update_frequency = update_frequency
        self.last_update = 0
        self.P_dis = None
        self.accuracy = None

    def _predict(self, opps):
        # we need to transform dataframe opps to structure as variable 'features' below in method _update_model here
        # or already in class Data
        to_be_predicted = transform(opps)
        self.P_dis = self.predictive_model.predict_proba(to_be_predicted)

    def _update_model(self):
        # It is assumed that in class Data() exist dataframe containing final data
        # HID, AID needs to be one-hot encoded or not used
        features = self.data.final_data[['HID', 'AID', 'H_diff_matches_1', '... some additional features']].to_numpy()
        labels = self.data.final_data[['H', 'D', 'A']].to_numpy()

        ####################################################
        #  HERE SET THE CLASSIFICATION MODEL'S PARAMETERS  #
        ####################################################
        clf = RandomForestClassifier(n_estimators=100)  # DEFAULT CLASSIFIER IS RANDOM FOREST
        if self.clf == 'xgb':
            clf = xgb.XGBClassifier()
        elif self.clf == 'ab':
            clf = AdaBoostClassifier()
        elif self.clf == 'gb':
            clf = GradientBoostingClassifier()
        elif self.clf == 'nb':
            clf = GaussianNB()
        elif self.clf == 'knn':
            clf = KNeighborsClassifier()
        elif self.clf == 'lr':
            clf = LogisticRegression()

        self.predictive_model_model = clf.fit(features, np.argmax(labels,
                                                                  axis=1))  # H has index 0, D has index 1, away has index 2
        ##################
        # FOR DEBUGGING  #
        #################
        if self.clf == 'rf':
            feature_importances = clf.feature_importances_
            std = np.std([tree.feature_importances_ for tree in clf.estimators_],
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
        self._predict(opps)
        return self.accuracy
