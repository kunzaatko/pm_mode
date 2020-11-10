import numpy as np
import pandas as pd

from models.elo import Elo
from bet_distribution.bet_distribution import Bet_distribution

elo = Elo()
bet_distribution = Bet_distribution()

class Model:
    def place_bets(self, opps, summary, inc):
        P_dis = elo.run_iter(inc, opps)

        if P_dis is not None:
            bet_distribution.run_iter(summary, opps, P_dis)
        else:
            bet_distribution.eval_opps(opps)

        return bet_distribution.bets

