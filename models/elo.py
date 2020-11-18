import numpy as np
import pandas as pd
import math
import sys

from models.feature_extraction.feature_extraction import Data

class Model_elo:
    def __init__(self, data,  mean_elo=1500, k_factor=20):
        '''
        Parameters:
        data(class): class for agregating and manipulating data for the problem.
        '''
        self.Data = data
        self.teams = self.Data.LL_data
        self.teams['ELO'] = np.NaN # založení column 'ELO'
        self.mean_elo = mean_elo
        self.k_factor = k_factor

    def __str__(self):
        return "Mean ELO: " + str(self.mean_elo) + "\n" + "K factor: " + str(self.k_factor) + "\n" + str(self.teams)

    def eval_update_ELOs(self, data_frame):
        '''
        Updates the ELOs for based on the games recorded in the `data_frame`. (mutates `self.teams['ELO']`)

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                        Has to include 'HID', 'AID', 'H', 'D' and 'A'.

        '''
        for (HID, AID, H, D, A) in data_frame[['HID','AID','H','D','A']].values:
            self.__update_ELO(HID, AID, (H,D,A))

    def __update_ELO(self, HID, AID, result):
        '''
        Updates the ELO for one match. This is the function to change if we want to change the algorithm. (mutates `self.teams['ELO']` `HID` and `AID`)

        Parameters:
        HID(int): Home team ID
        AID(int): Away team ID
        result(list): [home_win(bool/int), draw(bool/int), away_win(bool/int)]. The options are mutually exclusive.
        '''

        (home_win,_ , away_win) = result
        [home_elo, away_elo] = [self.teams.at[ID,'ELO'] for ID in [HID,AID]]

        [home_expected, away_expected] = [1/(1+10**((elo_1 - elo_2) / 400)) for (elo_1, elo_2) in [(away_elo, home_elo), (home_elo, away_elo)]]

        # Pokud někdo vyhrál, jelikož na remízy elo nefunguje
        if any([home_win, away_win]):
            self.teams.at[HID, 'ELO'] += self.k_factor * (home_win - home_expected)
            self.teams.at[AID, 'ELO'] += self.k_factor * (away_win - away_expected)

    def run_iter(self, inc, opps):
        '''
        Run the iteration of the evaluation loop.

        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                Has to include 'HID', 'AID', 'LID', 'H', 'D' and 'A'.
        opps(pandas.DataFrame): 'DataFrame' with the betting opportunities.
                                Has to include 'MatchID', 'HID' and 'AID'.
        Returns:
        pandas.DataFrame: 'DataFrame' loging the process of `P_dis_get` under this model.
        '''
        # Adding new teams
        for ID in self.Data.LL_data.index:
            if ID not in self.teams.index:
                self.teams.loc[ID] = self.Data.LL_data.loc[ID]
        self.teams.at[[math.isnan(d) for d in self.teams['ELO'].values],'ELO'] = self.mean_elo # define elo as mean elo for new teams

        self.eval_update_ELOs(inc)
        return self.P_dis_get(opps)

    def P_dis_get(self, data_frame):
        '''
        Calculate the probabilities based of match outcomes.

        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'

        Returns:
        pandas.DataFrame: 'DataFrame' loging the process of `P_dis_get` under this model.
        '''

        log = pd.DataFrame()

        P_dis = pd.DataFrame(columns=['P(H)', 'P(D)', 'P(A)'])

        for MatchID,(HID, AID) in zip(data_frame.index,data_frame[['HID','AID']].values):
            P_dis = P_dis.append(self.P_dis_match(MatchID,HID,AID))

        self.P_dis = P_dis

        return log

    def P_dis_match(self, MatchID, HID, AID):
        '''
        Calculate the probabitily of win lose draw of a match.

        Parameters:
        MatchID(int): The ID of the match. From the column 'MatchID' of the `data_frame`.
        HID(int): The ID of the home team. From the column 'HID' of the `data_frame`.
        AID(int): The ID of the home team. From the column 'AID' of the `data_frame`.

        Returns:
        pandas.DataFrame: 'DataFrame' with one row with the index `'MatchID'` and the associated outcome probabilities `'P(H)'`, `'P(D)'` and `'P(A)'`.
        '''

        [home_elo, away_elo] = [self.teams.at[ID,'ELO'] for ID in [HID,AID]]

        [home_expected, away_expected] = [1/(1+10**((elo_1 - elo_2) / 400)) for (elo_1, elo_2) in [(away_elo, home_elo), (home_elo, away_elo)]]

        return pd.DataFrame(data={'P(H)': [home_expected], 'P(D)': [0], 'P(A)': [away_expected]}, index=[MatchID])
