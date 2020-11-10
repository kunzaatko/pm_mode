import numpy as np
import pandas as pd

class Bet_distribution:
    def __init__(self):
        self.summary = None
        self.bets = None
        self.odds = None
        self.P_dis = None
        self.bet_exp_profit_margin = 1.05 # TODO: zkusit různé hodnoty abychom započítaly nepřesnost našeho modelu <08-11-20, kunzaatko> #

    def eval_summary(self, summary):
        '''
        Update `self.summary`.

        Parameters:
            summary(pandas.DataFrame):  Summary of the current iteration. (from env)
                                        Has to include 'Bankroll', 'max_bet' and 'min_bet'.
        '''
        self.summary = {'min_bet': summary.at[0, 'min_bet'], 'max_bet': summary.at[0, 'max_bet'], 'Bankroll': summary.at[0, 'Bankroll'] }

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

    def update_bets(self):
        '''
        Place optimal bets based `self.P_dis` and `self.odds`.
        '''
        # předpokládaný zisk na jeden vsazený kredit
        exp_profit = pd.DataFrame(data=(self.odds.to_numpy() * self.P_dis.to_numpy()), columns=["ExpH","ExpD","ExpA"], index=self.P_dis.index) # index sorted so we multiply matching elements

        # TODO: Tohle není deterministicky nejlepší možnost, ale počítá se vsazením co nejvíce peněz. <10-11-20, kunzaatko>
        while exp_profit.to_numpy().max() >= self.bet_exp_profit_margin:
            argmax = np.unravel_index(exp_profit.to_numpy().argmax(), exp_profit.to_numpy().shape)
            if self.summary['Bankroll'] >= (self.summary['min_bet'] + self.summary['max_bet']):
                bet = self.summary['max_bet']
                self.bets.iloc[argmax] = bet
                exp_profit.iloc[argmax] = 0 # abychom mohli najít další nejvyšší předpokládané zisky, vynulujeme zisky, na které už jsme vsadili
                self.summary['Bankroll'] -= bet

            elif self.summary['Bankroll'] >= 2*self.summary['min_bet']:
                bet = self.summary['Bankroll'] - self.summary['min_bet']
                self.bets.iloc[argmax] = bet
                exp_profit.iloc[argmax] = 0
                self.summary['Bankroll'] -= bet

            else:
                bet = self.summary['Bankroll']
                self.bets.iloc[argmax] = bet
                exp_profit.iloc[argmax] = 0
                self.summary['Bankroll'] -= bet
                break

    def run_iter(self, summary, opps, P_dis):
        '''
        The outermost API for Bet_distribution. Run bet_distribution on the iter.

        Parameters:
            summary(pandas.DataFrame):  Summary of the current iteration. (from env)
                                        Has to include 'Bankroll', 'max_bet' and 'min_bet'.
            opps(pandas.DataFrame): Opportunities for bets. (from env)
                                    Has to include 'OddsH','OddsD' and 'OddsA'.
            P_dis(pandas.DataFrame):    P_dis from model.
                                        Should include 'P(H)', 'P(D)' and 'P(A)'.
        '''
        self.eval_summary(summary)
        self.eval_opps(opps)
        self.eval_P_dis(P_dis)
        self.update_bets()

