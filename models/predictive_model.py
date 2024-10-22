import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt


class PredictiveModel(object):
    def __init__(self, data, classifier=RandomForestClassifier(), update_frequency=1, n_most_recent=2000,
                 use_recency=False, use_calibration=True, debug=False):
        """
        :param data: Data()
            Instance of class Data()
        :param classifier:
            Specifies the classifier supporting sklearn API
        :param update_frequency: int:
            Specifies how often to re-train model.
        :param n_most_recent: int:
            Specifies number of most recent matches which should be used to fit the model.
            This approach should speed up whole learning process but set right value to this attribute will be essential
        :param use_recency: bool:
            Specifies if use self.n_most_recent attribute to speed up training
        """
        self.data = data  # instance of class Data(), containing table which will be used as data to train
        self.update_frequency = update_frequency
        self.n_most_recent = n_most_recent  # this will be used simply as parameter to pd.DataFrame.tail() func
        self.use_recency = use_recency
        self.last_update = 0
        self.debug = debug

        self.clf = CalibratedClassifierCV(base_estimator=classifier, cv=5,
                                          method='sigmoid') if use_calibration else classifier
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = MinMaxScaler(feature_range=(0, 1), copy=True)  # scales cols of features between 0-1 (we can use
        # normalizer to normalize each row (input vector to one) instead.)
        self.pipeline = self.pipeline = Pipeline(
            steps=[('imputer', self.imputer), ('scaler', self.scaler),
                   ('classifier', self.clf)])  # HERE IS THE HEARTH OF MODEL (FITTED MODEL)

        self.P_dis = None
        self.accuracy = pd.DataFrame()  # TODO implement me if needed

    def _predict(self, opps):
        """
        Makes prediction and result stores in self.P_dis
        :param opps: pd.DataFrame:
            DataFrame containing opportunities for betting
        """
        # we need to transform dataframe opps to structure as variable 'features' below in method _update_model here
        # or already in class Data

        # thanks to xgboost sklearn API this should work for XGBClassifier too
        to_be_predicted = self.data.return_values().copy().to_numpy()  # this gives us 'fresh' opps containing also features for teams
        forecast = self.pipeline.predict_proba(to_be_predicted)
        self.P_dis = pd.DataFrame(columns=['P(H)', 'P(D)', 'P(A)'], data=forecast, index=opps.index)

    def _update_model(self):
        """
        Re-train the model stored in self.predictive_model
        """

        ##################
        # FIT THE MODEL  #
        ##################
        features = self.data.features.loc[:(self.data.opps_matches[0] - 1)].copy()
        labels = self.data.matches.loc[:(self.data.opps_matches[0] - 1), ['H', 'D', 'A']].copy()

        if self.use_recency:
            features = features.tail(self.n_most_recent)
            labels = labels.tail(self.n_most_recent)

        features = features.to_numpy()
        labels = labels.to_numpy()

        self.pipeline.fit(features, np.argmax(labels, axis=1))  # H has index 0, D has index 1, A has index 2

        ##################
        # FOR DEBUGGING  #
        ##################
        if self.debug:
            print(f"{self.clf.__class__.__name__} train accuracy: "
                  f"{self.pipeline.score(features, np.argmax(labels, axis=1))}")

            if hasattr(self.clf,
                       'feature_importances_'):  # FEATURES IMPORTANCES works only for tree based algorithms !!!
                # FEATURES IMPORTANCES # Warning: impurity-based feature importances can be misleading for high
                # cardinality features (many unique values). See sklearn.inspection.permutation_importance as an
                # alternative.
                feature_importances = self.clf.feature_importances_  # this will work only if
                # booster = gbtree !!!

                indices = np.argsort(feature_importances)[::-1]
                # Print the feature ranking
                print("Feature ranking based on feature_importances (MDI):")
                for f in range(features.shape[1]):
                    print(" %d. feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))

                # TODO try me as well
                # xgb.plot_importance(self.predictive_model)  # this requires matplotlib

    def test_me(self, test_features, test_labels):
        test_features = test_features.to_numpy()
        test_labels = test_labels.to_numpy()
        print(f"{self.clf.__class__.__name__} test accuracy: "
              f"{self.pipeline.score(test_features, np.argmax(test_labels, axis=1))}")
        print(f'classification report (test set): \n'
              f'{classification_report(np.argmax(test_labels, axis=1), np.argmax(self.P_dis.to_numpy(), axis=1), target_names=["H", "D", "A"], zero_division=0)}')
        # plot_confusion_matrix(self.pipeline, test_features, np.argmax(test_labels, axis=1), display_labels=["H", "D", "A"])
        # plt.show()
        cm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(self.P_dis.to_numpy(), axis=1),
                              labels=[0, 1, 2])
        print(f'Confusion matrix (test set): \n'
              f'{cm}')

    def tune_me(self, test_features, test_labels, hyperparams, type='rand'):
        """
        Applies cross-validation to find the best parameters
        :param test_features: pd.DataFrame:
        :param test_labels: pd.DataFrame:
        :param hyperparams: dict:
            Contains parameters which needs to be tested.
        :param type: str:
            Specifies if use gridsearch - ('grid') [tries all combinations specified in grid]
                             randomsearch - ('rand') [tries randomly possible values of parameters within given ranges]

        """

        grid = dict([('classifier__base_estimator__' + key, val) for key, val in hyperparams.items()]) if \
            self.clf.__class__.__name__ == 'CalibratedClassifierCV' else \
            dict([('classifier__' + key, val) for key, val in hyperparams.items()])
        test_features = test_features.to_numpy()
        test_labels = test_labels.to_numpy()

        search = GridSearchCV(self.pipeline, grid, cv=10, n_jobs=5) if type == 'grid' else \
            RandomizedSearchCV(self.pipeline, grid, cv=10, n_iter=50, n_jobs=5)
        print("Looking for best hyper-parameters settings...")
        search.fit(test_features, np.argmax(test_labels, axis=1))
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

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
