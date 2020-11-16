import numpy as np
import pandas as pd

class Bet_distribution:
    '''
    class for evaluating the distribution of the iteration Bankroll as bets in the betting opportunities.
    '''
    def __init__(self, exp_profit_margin = 1.05, max_bet=None):
        self.summary = None
        self.bets = None
        self.odds = None
        self.P_dis = None
        self.bet_exp_profit_margin = 1.05 # TODO: zkusit různé hodnoty abychom započítaly nepřesnost našeho modelu <08-11-20, kunzaatko> #
        self.max_bet=max_bet

        #  tested attribute
        self.matches_already_bet = set()

    def eval_summary(self, summary):
        '''
        Update `self.summary`.

        Parameters:
            summary(pandas.DataFrame):  Summary of the current iteration. (from env)
                                        Has to include 'Bankroll', 'Max_bet' and 'Min_bet'.
        '''
        if self.max_bet is None:
            self.summary = {'Min_bet': summary.at[0, 'Min_bet'], 'Max_bet': summary.at[0, 'Max_bet'], 'Bankroll': summary.at[0, 'Bankroll'] }
        else:
            self.summary = {'Min_bet': summary.at[0, 'Min_bet'], 'Max_bet': min(self.max_bet,summary.at[0,'Max_bet']), 'Bankroll': summary.at[0, 'Bankroll'] }

    def eval_opps(self, opps):
        '''
        Update `self.odds` and `self.bets`.

        Parameters:
            opps(pandas.DataFrame): Opportunities for bets. (from env)
                                    Has to include 'OddsH','OddsD' and 'OddsA'.
        '''
        self.odds = opps[['OddsH', 'OddsD', 'OddsA']].sort_index()
        self.bets = pd.DataFrame(data=np.zeros([len(opps),3]), columns=['BetH', 'BetD', 'BetA'], index=self.odds.index)

    def eval_P_dis(self, P_dis):
        '''
        Update `self.P_dis`

        Parameters:
            P_dis(pandas.DataFrame):    P_dis from model.
                                        Should include 'P(H)', 'P(D)' and 'P(A)'.
        '''
        self.P_dis = P_dis.sort_index()

    def optimize(self, exp, opps):
        """
        Approach based on this work:
        [https://www.researchgate.net/publication/277284931_Predicting_and_Retrospective_Analysis_of_Soccer_Matches_in_a_League]

        :param exp: pd.DataFrame:
            Containing expected profits while betting one unit
        :param opps: pd.DataFrame:
            Containing current opportunities for betting
        :return: np.ndarray:
            Matrix with same shape as opps containing optimal bets (some of them can be higher than 'Max_bet' and )
        """
        exp_profit_matches = (exp["ExpH"] + exp["ExpD"] + exp["ExpA"]).to_numpy()  # E[profit for match]
        opp = opps.to_numpy()
        opt = []
        for i in np.arange(exp_profit_matches.size):
            vec = opp[i] / (2.0 * np.power((opp[i] - exp[i]), 2))
            opt.append(vec)
        return np.stack(opt)

    def kelly_criterion(self, exp, opps):
        """
        Approach based on Kelly criterion.
        :param exp: pd.DataFrame:
            Containing expected profits while betting one unit
        :param opps: pd.DataFrame:
            Containing current opportunities for betting
        :return: np.ndarray:
            Matrix with same shape as opps containing optimal bets. Values means how much percent of bankroll to stake.
        """
        kelly = (exp.to_numpy() - 1) / (opps.to_numpy() - 1)
        return np.where(kelly <= 0.0, 0.0, kelly)  # negative values means non-positive expected profit so it is set to 0

    def eval_inc(self, inc):
        """
        Removes teams from self.matches_already_bet if present in current inc
        :param inc: pd.DataFrame:
            Current increment of data
        """
        vals = inc.index.values.astype(int)
        for val in vals:
            if val in self.matches_already_bet:
                self.matches_already_bet.remove(val)

    def update_bets(self):
        '''
        Place optimal bets based `self.P_dis` and `self.odds`.

        Returns:
        pd.DataFrame: log of the bet distibution process.
        '''
        log = pd.DataFrame()
        # předpokládaný zisk na jeden vsazený kredit
        exp_profit = pd.DataFrame(data=(self.odds.to_numpy() * self.P_dis.to_numpy()), columns=["ExpH","ExpD","ExpA"], index=self.P_dis.index) # index sorted so we multiply matching elements
        #opt = self.optimize(exp_profit, self.odds)
        kelly = self.kelly_criterion(exp_profit, self.odds)
        # TODO: Tohle není deterministicky nejlepší možnost, ale počítá se vsazením co nejvíce peněz. <10-11-20, kunzaatko>
        while exp_profit.to_numpy().max() >= self.bet_exp_profit_margin:
            argmax = np.unravel_index(exp_profit.to_numpy().argmax(), exp_profit.to_numpy().shape)
            #ind = self.bets.index.values.astype(int)[argmax[0]]
            #if ind not in self.matches_already_bet:
            #    self.matches_already_bet.add(ind)
            if self.summary['Bankroll'] >= (self.summary['Min_bet'] + self.summary['Max_bet']):
                bet_0 = kelly[argmax] * self.summary['Bankroll'] * 0.33
                bet = bet_0 if self.summary['Min_bet'] <= bet_0 <= self.summary['Max_bet'] else\
                    self.summary['Max_bet'] if bet_0 >= self.summary['Max_bet'] else self.summary['Min_bet']
                self.bets.iloc[argmax] = bet
                self.summary['Bankroll'] -= bet

            elif self.summary['Bankroll'] >= 2*self.summary['Min_bet']:
                bet = self.summary['Bankroll'] - self.summary['Min_bet']
                self.bets.iloc[argmax] = bet
                self.summary['Bankroll'] -= bet

            else:
                bet = self.summary['Bankroll']
                self.bets.iloc[argmax] = bet
                self.summary['Bankroll'] -= bet
                break
            exp_profit.iloc[argmax] = 0

        # TODO: Now returning empty log. Add something to the log to return. <11-11-20, kunzaatko> #
        return log

    def run_iter(self, summary, opps, P_dis):
        '''
        The outermost API for Bet_distribution. Run bet_distribution on the iter.

        Parameters:
            summary(pandas.DataFrame):  Summary of the current iteration. (from env)
                                        Has to include 'Bankroll', 'Max_bet' and 'Min_bet'.
            opps(pandas.DataFrame): Opportunities for bets. (from env)
                                    Has to include 'OddsH','OddsD' and 'OddsA'.
            P_dis(pandas.DataFrame):    P_dis from model.
                                        Should include 'P(H)', 'P(D)' and 'P(A)'.
        '''
        self.eval_summary(summary)
        self.eval_opps(opps)
        self.eval_P_dis(P_dis)
        self.update_bets()
        # this should remove teams from self.matches_already_bet if present in inc, inc param have to be added to run_iter func
        #self.eval_inc(inc)

