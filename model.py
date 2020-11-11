import numpy as np
import pandas as pd

from models.elo import Elo
from bet_distribution.bet_distribution import Bet_distribution

bet_distribution = Bet_distribution
model = Elo
model_params={}
bet_distribution_params={}

class Model:
    def __init__(self, model=model, model_params=model_params, log=True, bet_distribution = bet_distribution, bet_distribution_params={}):
        '''
        Initialization of the model class with the parameters we want to use for evaluation.

        Parameters:
        method()
        log(bool): whether to log the process. If set to false, then self.log is false. Else is `self.log = (log_model, log_bet_distribution)`
        '''

        self.model = model(**model_params)
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

        log_model = self.model.run_iter(inc, opps)

        log_bet_distribution = self.bet_distribution.run_iter(summary, opps, self.model.P_dis)

        if self.log is True:
            self.log = (log_model, log_bet_distribution)

        return self.bet_distribution.bets

