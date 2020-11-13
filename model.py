import numpy as np
import pandas as pd

from models.elo import Model_elo
from bet_distribution.bet_distribution import Bet_distribution
from models.feature_extraction.feature_extraction import Data

if 'bet_distribution' not in locals():
    bet_distribution = Bet_distribution
if 'model' not in locals():
    model = Model_elo
if 'model_params' not in locals():
    model_params={}
if 'bet_distribution_params' not in locals():
    bet_distribution_params={}

class Model:
    def __init__(self, model=model, model_params=model_params, log=True, bet_distribution = bet_distribution, bet_distribution_params={}):
        '''
        Initialization of the model class with the parameters we want to use for evaluation.

        Parameters:
        model(class): `class` that represents the model used. It has to include the attribute `model.P_dis` and has to have the method `model.run_iter(inc,opps)` that returns the log. It is read from the `model` local variable.
        model_params(dict): A dictionary of params to pass to the `model`. It is read from the `model_params` local variable.
        log(bool): Whether to log the process. If set to `false`, then `self.log` is `false`. Else is `self.log = (log_model, log_bet_distribution)`. Where` log_model` is the log that `model.run_iter(...)` returns and `bet_distribution.run_iter(...)` returns.
        bet_distribution_params(dict): A dictionary of params to pass to the `bet_distribution`. It is read from the `bet_distribution_params` local variable.
        '''

        self.data = Data()
        self.model = model(self.data, **model_params)
        self.bet_distribution = bet_distribution(**bet_distribution_params)
        self.log = log


    def place_bets(self, opps, summary, inc):
        '''
        The outermost API method for the evaluation loop. The evaluation loop relies on the avalibility of this method for the model class.

        Parameters:
        All the parameters are supplied by the evaluation loop.
        opps(pandas.DataFrame): dataframe that includes the opportunities for betting.
        summary(pandas.DataFrame): includes the `Max_bet`, `Min_bet` and `Bankroll`.
        inc(pandas.DataFrame): includes the played matches with the scores for the model.

        Returns:
        pandas.DataFrame: With the bets that we want to place. Indexed by the teams `ID`.
        '''


        self.data.update_data(opps=opps,summary=summary, inc=inc)

        log_model = self.model.run_iter(inc, opps)

        self.data.update_data(P_dis=self.model.P_dis)

        log_bet_distribution = self.bet_distribution.run_iter(summary, opps, self.model.P_dis)
        self.data.update_data(bets=self.bet_distribution.bets)

        if self.log is True:
            self.log = (log_model, log_bet_distribution)

        self.data.end_data_agregation_iter()

        return self.bet_distribution.bets

