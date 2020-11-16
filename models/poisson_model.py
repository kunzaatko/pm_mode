import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


class PoissonRegression(object):
    """
    Class wraps basic time independent Poisson model based on predictions of outcome scores of match.
    Number of goals which both teams will score in match are assumed to be independent random variables. This is not
    true in reality. Models then  calculates parameters representing attack and defense strengths for each team and
    model also includes home advantage parameter.
    """
    def __init__(self, data, update_frequency=2):
        self.data = data
        self.goal_data = pd.DataFrame()
        self.teams = set()
        self.model = None
        self.P_dis = None  # most recent P_dis
        self.last_update = 0
        self.update_frequency = update_frequency
        self.accuracy = pd.DataFrame()

    def _update_model(self):
        """
        Creates/updates time independent Poisson regression model based on actual goal data.
        :return:
            Returns fitted time independent poisson regression model.
        """
        self.model = smf.glm(formula="goals ~ home + C(team) + C(opponent)", data=self.goal_data,
                             family=sm.families.Poisson()).fit_regularized(L1_wt=0, alpha=0.01)

    def _update_teams(self, inc):
        """
        Updates set of teams already used to fit.
        :param inc: pd.DataFrame:
            Has to include 'HID', 'AID' columns,
            but it is assumed to contain 'LID', 'H', 'D', 'A' ... columns as well.
        """
        self.teams.update(pd.concat([inc["HID"], inc["AID"]]).unique())

    def _update_goal_data(self, inc):
        """
        Updates 'self.goal_data' DataFrame based on new data contained in 'inc' parameter
        :param inc: pd.DataFrame:
            Has to include 'HID', 'AID', 'HSC', 'ASC' columns.
        """
        new_data = pd.concat([inc[['HID', 'AID', 'HSC']].assign(home=1).rename(
            columns={'HID': 'team', 'AID': 'opponent', 'HSC': 'goals'}),
            inc[['AID', 'HID', 'ASC']].assign(home=0).rename(
                columns={'AID': 'team', 'HID': 'opponent', 'ASC': 'goals'})])
        self.goal_data = pd.concat([self.goal_data, new_data])

    def _simulate_match(self, match, max_goals=10):
        """
        Simulates match based on model and predicts probabilities of W/D/L for homeTeam(HID).
        :param match: pd.Series (row of DataFrame):
            Has to include 'HID', 'AID'
        :param max_goals: int:
            The maximum number of goals that we assume will occur.
        :return: np.array:
            [P(H), P(D), P(A)]
        """
        home_goals_avg = self.model.predict(pd.DataFrame(data={'team': match["HID"],
                                                               'opponent': match["AID"], 'home': 1},
                                                         index=[1])).values[0]
        away_goals_avg = self.model.predict(pd.DataFrame(data={'team': match["AID"],
                                                               'opponent': match["HID"], 'home': 0},
                                                         index=[1])).values[0]
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                     [home_goals_avg, away_goals_avg]]
        goals = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

        return np.array([np.sum(np.tril(goals, -1)), np.sum(np.diag(goals)), np.sum(np.triu(goals, 1))])

    def _make_prediction(self, opps):
        """
        Predicts the probabilities of outcome [P(H), P(D), P(A)] of matches given in increment
        :param opps: pd.DataFrame:
            Has to contain 'HID', 'AID' column.
        :return: pd.DataFrame:
            One row of DataFrame looks like:
                                    row([P(H), P(D), P(A)]) - if both teams in match was previously 'seen' in fit phase
                                    row([0., 0., 0.]) - otherwise
        """
        predictions = [np.array([0., 0., 0.]) if row["HID"] not in self.teams or row["AID"] not in self.teams else
                       self._simulate_match(row) for idx, row in opps.iterrows()]
        return pd.DataFrame(data=predictions, columns=["P(H)", "P(D)", "P(A)"], index=opps.index)

    def _eval_inc(self, inc):
        """
        Evaluates data increment:
            1) Adds previously unknown teams to 'self.teams'
            2) Updates 'self.goal_data'
            3) Fit the model on new data in 'self.goal_data'
        :param inc: pd.DataFrame:
            Has to include 'HID', 'AID', 'HSC', 'ASC' columns.
        """
        self._update_teams(inc)
        self._update_goal_data(inc)
        if self.last_update % self.update_frequency == 0:
            self._update_model()
        self.last_update += 1

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
        self._eval_inc(inc)
        self.P_dis = self._make_prediction(opps)
        #self._evaluate_accuracy(opps)
        return self.accuracy

    def _evaluate_accuracy(self):
        """
        Calculates accuracy.
        TODO calculate accuracy based on attributes stored in class Data in attribute self.data
        :return:
        """
        pass
