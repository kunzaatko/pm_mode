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

    def update_bets(self):
        '''
        Place optimal bets based `self.P_dis` and `self.odds`.
        '''
        logs = pd.DataFrame()
        # předpokládaný zisk na jeden vsazený kredit
        exp_profit = pd.DataFrame(data=(self.odds.to_numpy() * self.P_dis.to_numpy()), columns=["ExpH","ExpD","ExpA"], index=self.P_dis.index) # index sorted so we multiply matching elements

        # TODO: Tohle není deterministicky nejlepší možnost, ale počítá se vsazením co nejvíce peněz. <10-11-20, kunzaatko>
        while exp_profit.to_numpy().max() >= self.bet_exp_profit_margin:
            argmax = np.unravel_index(exp_profit.to_numpy().argmax(), exp_profit.to_numpy().shape)
            if self.summary['Bankroll'] >= (self.summary['Min_bet'] + self.summary['Max_bet']):
                bet = self.summary['Max_bet']
                self.bets.iloc[argmax] = bet
                exp_profit.iloc[argmax] = 0 # abychom mohli najít další nejvyšší předpokládané zisky, vynulujeme zisky, na které už jsme vsadili
                self.summary['Bankroll'] -= bet

            elif self.summary['Bankroll'] >= 2*self.summary['Min_bet']:
                bet = self.summary['Bankroll'] - self.summary['Min_bet']
                self.bets.iloc[argmax] = bet
                exp_profit.iloc[argmax] = 0
                self.summary['Bankroll'] -= bet

            else:
                bet = self.summary['Bankroll']
                self.bets.iloc[argmax] = bet
                exp_profit.iloc[argmax] = 0
                self.summary['Bankroll'] -= bet
                break

            # TODO: Now returning empty log. Add something to the log to return. <11-11-20, kunzaatko> #
            return logs


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

