import pandas as pd
from scipy.stats import poisson, skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from numba import jit, njit

dataset = pd.read_csv('../training_data.csv', parse_dates=['Date', 'Open'])


def poisson_time_independent(increment, older_relevant=pd.DataFrame()):
    """
    Based on older data and new increment data it creates time independent Poisson regression model.
    :param increment: pd.DataFrame
    :param older_relevant: pd.DataFrame
    :return:
        Returns fitted time independent poisson regression model.
    """
    data = pd.concat([older_relevant, increment])
    goal_data = pd.concat([data[['HID', 'AID', 'HSC']].assign(home=1).rename(
        columns={'HID': 'team', 'AID': 'opponent', 'HSC': 'goals'}),
        data[['AID', 'HID', 'ASC']].assign(home=0).rename(
            columns={'AID': 'team', 'HID': 'opponent', 'ASC': 'goals'})])

    return smf.glm(formula="goals ~ home + C(team) + C(opponent)", data=goal_data,
                   family=sm.families.Poisson()).fit()


def simulate_match(model, match, max_goals=10):
    """
    Simulates match based on model and predicts probabilities of W/D/L for homeTeam.
    :param match: pd.Series (row of dataframe representing match)
    :param model:
    :param max_goals:
    :return: np.array
        if home/away team was in training data then probabilities of W/D/L of homeTeam otherwise [0., 0., 0.]
    """

    home_goals_avg = model.predict(pd.DataFrame(data={'team': match["HID"],
                                                      'opponent': match["AID"], 'home': 1},
                                                index=[1])).values[0]
    away_goals_avg = model.predict(pd.DataFrame(data={'team': match["AID"],
                                                      'opponent': match["HID"], 'home': 0},
                                                index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals_avg, away_goals_avg]]
    goals = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

    return np.array(
        [match["Unnamed: 0"], np.sum(np.tril(goals, -1)), np.sum(np.diag(goals)), np.sum(np.triu(goals, 1))])

@jit
def predict(teams_used_to_fit, increment, model):
    predictions = []
    for idx, row in increment.iterrows():
        if row["HID"] not in teams_used_to_fit or row["AID"] not in teams_used_to_fit:
            predictions.append(np.array([idx, 0., 0., 0.]))
        else:
            predictions.append(simulate_match(model, row))
    return np.stack(predictions)


def evaluate_accuracy(prediction, increment):
    """

    :param prediction: np.ndarray
        [[Match_ID, P(H), P(D), P(A)] X num_of_matches]
    :param increment: pd.DataFrame
    :return: tuple
        (num_of_correctly_predicted/num_of_all - num_of_not_present, num_of_correctly_predicted/num_of_all)
    """
    truemap = (increment[['H', 'D', 'A']]).to_numpy()
    prediction = prediction[:, 1:]
    max = (prediction == prediction.max(axis=1)[:, None]).astype(int)
    missing = max.all(axis=1)
    max = max[~missing]
    truemap = truemap[~missing]
    result = np.multiply(max, truemap).any(axis=1)

    return len(np.where(result == True)[0]) / result.size, len(np.where(result == True)[0]) / prediction.shape[0]


if __name__ == "__main__":
    import time
    seasons = [dataset[dataset['Sea'] == sea] for sea in dataset['Sea'].unique()]
    new = pd.concat([seasons[i] for i, sea in enumerate(seasons) if i != 0])
    increments = [new[new['Date'] == sea] for sea in new['Date'].unique()]

    # initially fit the model with first season, main problem is that not all teams used to test model are present in
    # train data
    poisson_model = poisson_time_independent(increment=seasons[0])
    relevant = seasons[0]
    teams_used = pd.concat([relevant["HID"], relevant["AID"]]).unique()

    for increment in seasons[1:]:

        t0 = time.time()
        prediction = predict(teams_used, increment, poisson_model)
        print(f"Prediction evaluated in {time.time()-t0} s")
        print(f"Accuracies {evaluate_accuracy(prediction, increment)} \n\n")

        # update model, teams used and relevant data within new increment
        poisson_model = poisson_time_independent(increment, older_relevant=relevant)
        teams_used = pd.concat([relevant["HID"], relevant["AID"], increment["HID"], increment["AID"]]).unique()
        relevant = pd.concat([relevant, increment])
        input("Type anything to continue")
