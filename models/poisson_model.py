import pandas as pd
from scipy.stats import poisson, skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from numba import jit, njit

dataset = pd.read_csv('../training_data.csv', parse_dates=['Date', 'Open'])


def poisson_time_independent(inc, older_relevant=pd.DataFrame()):
    """
    Creates time independent Poisson regression model based on older_data and new increment of data.
    :param inc: pd.DataFrame
    :param older_relevant: pd.DataFrame
    :return:
        Returns fitted time independent poisson regression model.
    """
    data = pd.concat([older_relevant, inc])
    goal_data = pd.concat([data[['HID', 'AID', 'HSC']].assign(home=1).rename(
        columns={'HID': 'team', 'AID': 'opponent', 'HSC': 'goals'}),
        data[['AID', 'HID', 'ASC']].assign(home=0).rename(
            columns={'AID': 'team', 'HID': 'opponent', 'ASC': 'goals'})])

    return smf.glm(formula="goals ~ home + C(team) + C(opponent)", data=goal_data,
                   family=sm.families.Poisson()).fit_regularized(L1_wt=0, alpha=0.01)


def simulate_match(model, match, max_goals=10):
    """
    Simulates match based on model and predicts probabilities of W/D/L for homeTeam.
    :param match: pd.Series
        (row of dataframe representing match)
    :param model:
    :param max_goals: int
        max number of assumed goals the can score
    :return: np.array
        [P(H), P(D), P(A)]
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

    return np.array([np.sum(np.tril(goals, -1)), np.sum(np.diag(goals)), np.sum(np.triu(goals, 1))])


def predict(teams_used_to_fit, inc, model):
    """
    Predicts the probabilities of outcome [H, D, A] of matches given in increment
    :param teams_used_to_fit: list/np.array
        all previously "seen" teams used for fit the model
    :param inc: pd.Dataframe
        dataframe containing matches to predict
    :param model:
    :return: pd.Dataframe
        row([P(H), P(D), P(A)]) if both teams in match was previously seen otherwise row([0., 0., 0.])
    """
    predictions = []
    for idx, row in inc.iterrows():
        if row["HID"] not in teams_used_to_fit or row["AID"] not in teams_used_to_fit:
            predictions.append(np.array([0., 0., 0.]))
        else:
            predictions.append(simulate_match(model, row))
    return pd.DataFrame(data=predictions, columns=["H", "D", "A"], index=inc.index)


def evaluate_accuracy(pred, inc):
    """
    Evaluates accuracy.
    :param pred: np.ndarray
        [[Match_ID, P(H), P(D), P(A)] X num_of_matches]
    :param inc: pd.DataFrame
    :return: tuple
        (num_of_correctly_predicted/(num_of_all - num_of_not_present), num_of_correctly_predicted/num_of_all)
    """
    truemap = (inc[['H', 'D', 'A']]).to_numpy()
    pred = pred[['H', 'D', 'A']].to_numpy()
    max = (pred == pred.max(axis=1)[:, None]).astype(int)
    missing = max.all(axis=1)
    max = max[~missing]
    truemap = truemap[~missing]
    result = np.multiply(max, truemap).any(axis=1)

    return len(np.where(result == True)[0]) / result.size, len(np.where(result == True)[0]) / prediction.shape[0]


if __name__ == "__main__":

    seasons = [dataset[dataset['Sea'] == sea] for sea in dataset['Sea'].unique()]
    new = pd.concat([seasons[i] for i, sea in enumerate(seasons) if i > 4])  # it is fitted on first 3 seasons
    increments = [new[new['Date'] == sea] for sea in new['Date'].unique()]

    # initially fit the model on first 5 seasons
    relevant = pd.concat([seasons[0], seasons[1], seasons[2], seasons[3], seasons[4]])
    poisson_model = poisson_time_independent(inc=relevant)
    teams_used = pd.concat([relevant["HID"], relevant["AID"]]).unique()

    for increment in increments:  # use seasons[5:] to obtain more relevant test of accuracy

        prediction = predict(teams_used, increment, poisson_model)
        print(prediction)
        print(increment[["Sea", "OddsH", "OddsD", "OddsA", "HSC", "ASC"]])

        acc = evaluate_accuracy(prediction, increment)
        print(f"Accuracy of match outcome calculated considering only previously seen teams {acc[0]} \n"
              f"Accuracy calculated on all matches {acc[1]}\n")

        # update model, teams used and relevant data within new increment
        poisson_model = poisson_time_independent(increment, older_relevant=relevant)
        relevant.append(increment)
        teams_used = pd.concat([relevant["HID"], relevant["AID"]]).unique()
        input("Type anything to continue")
