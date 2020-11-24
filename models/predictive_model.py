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


class PredictiveModel(object):
    def __init__(self, data, classifier='rf', update_frequency=1, n_most_recent=2000, use_recency=False):
        """
        :param data: Data()
            Instance of class Data()
        :param classifier: str:
            Specifies the classifier used to predictions:
                        'rf' - Random forest
                        'ab' - Adaptive boosting
                        'gb' - Gradient boosting
                        'xgb' - XGBoost (Extreme gradient boosting)
                        'gnb' - Gaussian NaiveBayes
                        'knn' - K-nearest-neighbours
                        'lr' - Logistic regression
                        'vc' - Voting classifier
        :param update_frequency: int:
            Specifies how often to re-train model.
        :param n_most_recent: int:
            Specifies number of most recent matches which should be used to fit the model.
            This approach should speed up whole learning process but set right value to this attribute will be essential
        :param use_recency: bool:
            Specifies if use self.n_most_recent attribute to speed up training
        """
        self.data = data  # instance of class Data(), containing table which will be used as data to train
        self.clf = classifier  # specifies type of model
        self.update_frequency = update_frequency
        self.n_most_recent = n_most_recent  # this will be used simply as parameter to pd.DataFrame.tail() func
        self.use_recency = use_recency
        self.last_update = 0

        self.predictive_model = RandomForestClassifier(n_estimators=100)
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = MinMaxScaler(feature_range=(0, 1), copy=True)  # scales cols of features between 0-1 (we can use
        # normalizer to normalize each row (input vector to one) instead.)
        self.pipeline = None  # HERE IS THE HEARTH OF MODEL (FITTED MODEL)

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

        ##########################################
        #  HERE SET THE CLASSIFIER'S PARAMETERS  #
        ##########################################
        self.predictive_model = RandomForestClassifier(n_estimators=100,  # DEFAULT CLASSIFIER IS RANDOM FOREST
                                                       criterion='gini',
                                                       max_depth=6,
                                                       min_samples_split=2,
                                                       min_samples_leaf=1,
                                                       max_features='auto',
                                                       bootstrap=True,
                                                       oob_score=False,
                                                       n_jobs=None,
                                                       random_state=None,
                                                       warm_start=False,
                                                       # When warm_start is true, the existing fitted model
                                                       # attributes are used to initialize the new model in a
                                                       # subsequent call to fit.
                                                       class_weight=None,
                                                       max_samples=None)
        if self.clf == 'xgb':
            # What makes XGBoost unique is that it uses “a more regularized model formalization to control
            # over-fitting, which gives it better performance” , according to the author of the algorithm,
            # Tianqi Chen. Therefore, it helps to reduce overfitting. TODO test Standalone Random Forest With
            #  Scikit-Learn-Like API (https://xgboost.readthedocs.io/en/latest/tutorials/rf.html)
            self.predictive_model = xgb.XGBClassifier(n_estimators=100,  # TODO consider incremental training
                                                      max_depth=6,
                                                      # Increasing this value will make the model more complex and
                                                      # more likely to overfit Beware that XGBoost aggressively
                                                      # consumes memory when training a deep tree.
                                                      learning_rate=0.1,
                                                      verbosity=1,
                                                      objective='multi:softmax',
                                                      num_class=3,
                                                      booster='gbtree',
                                                      tree_method='auto',
                                                      # It’s recommended to study this option from parameters document.
                                                      # The tree construction algorithm used in XGBoost. See description
                                                      # https://arxiv.org/pdf/1603.02754.pdf
                                                      n_jobs=None,
                                                      gamma=0,
                                                      # The larger gamma is, the more conservative the algorithm will
                                                      # be. range: [0,∞]
                                                      min_child_weight=1,  # range: [0,∞]
                                                      subsample=1,
                                                      reg_alpha=0,
                                                      # L1 regularization term on weights. Increasing this value will
                                                      # make model more conservative. AKA Lasso regularization
                                                      reg_lambda=1,
                                                      # L2 regularization term on weights. Increasing this value will
                                                      # make model more conservative. AKA Ridge regularization
                                                      scale_pos_weight=1,
                                                      base_score=0.5,
                                                      random_state=None,
                                                      missing=None,
                                                      # maybe not use this because we have to process this more
                                                      # generally
                                                      num_parallel_tree=3,
                                                      monotone_constraints=None,
                                                      interaction_constraints=None,
                                                      importance_type='gain')
        elif self.clf == 'ab':
            self.predictive_model = AdaBoostClassifier(base_estimator=None,
                                                       n_estimators=50,
                                                       learning_rate=1,
                                                       algorithm='SAMME.R',
                                                       random_state=None)
        elif self.clf == 'gb':
            self.predictive_model = GradientBoostingClassifier(loss='deviance',
                                                               learning_rate=0.1,
                                                               n_estimators=100,
                                                               subsample=1.0,
                                                               criterion='friedman_mse',
                                                               min_samples_split=2,
                                                               min_samples_leaf=1,
                                                               max_depth=6,
                                                               init=None,
                                                               # init has to provide fit and predict_proba. If
                                                               # ‘zero’, the initial raw predictions are set to zero.
                                                               # By default, a DummyEstimator predicting the classes
                                                               # priors is used.
                                                               random_state=None,
                                                               max_features=None,
                                                               verbose=0,
                                                               warm_start=False,
                                                               # When warm_start is true, the existing fitted model
                                                               # attributes are used to initialize the new model in a
                                                               # subsequent call to fit.
                                                               validation_fraction=0.1,
                                                               n_iter_no_change=None,
                                                               )
        elif self.clf == 'vc':
            self.predictive_model = VotingClassifier(estimators=[],
                                                     voting='hard',
                                                     weights=None,
                                                     n_jobs=None,
                                                     verbose=False)
        elif self.clf == 'gnb':
            self.predictive_model = GaussianNB()
        elif self.clf == 'knn':
            # The algorithm gets significantly slower as the number of examples and/or predictors/independent
            # variables increase.
            self.predictive_model = KNeighborsClassifier(n_neighbors=5,
                                                         weights='uniform',
                                                         algorithm='auto',
                                                         leaf_size=30,
                                                         p=2,
                                                         # Power parameter for the Minkowski metric. p=1 (l1),
                                                         # p=2 (l2), p (lp)
                                                         metric='minkowski',
                                                         n_jobs=None)
        elif self.clf == 'lr':
            self.predictive_model = LogisticRegression(penalty='l2',
                                                       C=1.0,
                                                       class_weight=None,
                                                       random_state=None,
                                                       solver='lbfgs',
                                                       max_iter=100,
                                                       multi_class='auto',
                                                       verbose=0,
                                                       warm_start=False,
                                                       # When warm_start is true, the existing fitted model
                                                       # attributes are used to initialize the new model in a
                                                       # subsequent call to fit.
                                                       n_jobs=None)

        ##################
        # FIT THE MODEL  #
        ##################
        # It is assumed that in class Data() exist dataframe containing final data
        # HID, AID needs to be one-hot encoded or not used
        features = self.data.features.copy()
        labels = self.data.matches[['H', 'D', 'A']].copy()

        if self.use_recency:
            features = features.tail(self.n_most_recent)
            labels = labels.tail(self.n_most_recent)

        features = features.to_numpy()
        labels = labels.to_numpy()

        self.pipeline = Pipeline(
            steps=[('imputer', self.imputer), ('scaler', self.scaler), ('classifier', self.predictive_model)])
        # features = self.imputer.fit_transform(features)  # impute missing (np.nan) values
        # features = self.scaler.fit_transform(features)  # scale the feature columns between 0-1
        self.pipeline.fit(features, np.argmax(labels, axis=1))  # H has index 0, D has index 1, A has index 2

        ##################
        # FOR DEBUGGING  #
        ##################
        print(f"{self.predictive_model.__class__.__name__} train accuracy: "
              f"{self.pipeline.score(features, np.argmax(labels, axis=1))}")

        if self.clf in ['rf', 'ab', 'xgb', 'gb']:  # FEATURES IMPORTANCES works only for tree based algorithms !!!
            # FEATURES IMPORTANCES # Warning: impurity-based feature importances can be misleading for high
            # cardinality features (many unique values). See sklearn.inspection.permutation_importance as an
            # alternative.
            feature_importances = self.predictive_model.feature_importances_  # this will work only if
                                                                              # booster = gbtree !!!
            std = np.std([tree.feature_importances_ for tree in self.predictive_model.estimators_],
                         axis=0)
            indices = np.argsort(feature_importances)[::-1]
            # Print the feature ranking
            print("Feature ranking based on feature_importances (MDI):")
            for f in range(features.shape[1]):
                print(" %d. feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))

            # TODO try me as well
            # xgb.plot_importance(self.predictive_model)  # this requires matplotlib

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
