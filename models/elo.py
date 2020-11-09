#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ELO:
    def __init__(self, mean_elo=1500, k_factor=20):
        # frame of teams that are in the league
        self.teams = pd.DataFrame(columns=["LIDs", "ELO"]) # a team can be in multiple leagues therefore 'LIDs' and not 'LID' viz. '../scripts/multiple_LIDs_for_one_team.py' # TODO: rozdělit ELO podle ligy <09-11-20, kunzaatko> #
        self.mean_elo = mean_elo
        self.k_factor = k_factor

    def __str__(self):
        return "Mean ELO: " + str(self.mean_elo) + "\n" + "K factor: " + str(self.k_factor) + "\n" + "Draw factor: " + str(self.draw_factor) + "\n" + str(self.teams)

    def eval_opps(self, opps):
        '''
        Evaluate betting opportunities:
            1) Adds previously unknown teams to `self.teams`
            2) Adds LIDs new for the teams to `self.teams`
        '''
        self.eval_new_teams(opps)
        self.eval_new_LIDs(opps)

    def eval_inc(self, inc):
        '''
        Evaluate data increment:
            1) Adds previously unknown teams to `self.teams`
            2) Adds LIDs new for the teams to `self.teams`
            3) Evaluates the new ELOs for the teams
        '''
        self.eval_new_teams(inc)
        self.eval_new_LIDs(inc)
        self.eval_update_ELOs(inc)


    def eval_new_teams(self, data_frame):
        '''
        New teams in `data_frame` to `self.teams` and associate `self.mean_elo` with them. (appends to `self.teams`)

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        # FIXME: This could be done in one run through cating the 'HID' and the 'AID' cols <09-11-20, kunzaatko> #
        new_teams = data_frame[[home_team not in self.teams.index for home_team in data_frame['HID']]]['HID'].append(data_frame[[ away_team not in self.teams.index for away_team in data_frame['AID']]]['AID']).unique()

        for team in new_teams:
            self.teams = self.teams.append(pd.DataFrame(data={'LIDs': [[]], 'ELO': [self.mean_elo]},index=[team]))

    def eval_new_LIDs(self, data_frame):
        '''
        If team is playing in a league that it did not play before, associate the 'LID' with it. (mutates `self.teams['LIDs']`)

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new LIDs from for the teams in `self.teams` (`inc` and `opps`).
                                        Has to include 'HID', 'AID' and 'LID'.
        '''
        for team in self.teams.index:
            # FIXME: This could be done in one run through cating the 'HID' and the 'AID' cols <09-11-20, kunzaatko> #
            for LID in data_frame.set_index('HID').at[team, 'LID'].append(data_frame.set_index('AID').at[team, 'LID']).unique():
                if LID not in self.teams.at[team,'LIDs']:
                    self.teams.at[team, 'LIDs'].append(LID)

    def eval_update_ELOs(self, data_frame):
        '''
        Updates the ELOs for based on the games recorded in the `data_frame`. (mutates `self.teams['ELO']`)

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                        Has to include 'HID', 'AID', 'H', 'D' and 'A'.

        '''
        for (HID, AID, H, D, A) in data_frame[['HID','AID','H','D','A']].values:
            self.update_ELO(HID, AID, (H,D,A))

    def update_ELO(self, HID, AID, result):
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
        pandas.DataFrame or None: 'DataFrame' indexed by `'MatchID'`s and the associated outcome probabilities `'P(H)'`, `'P(D)'` and `'P(A)'` for all matches in `opps` or `None` if no `opps` where passed.
        '''
        if inc is not None:
            self.eval_inc(inc)

        if opps is not None:
            self.eval_opps(opps)
            return self.P_dis(opps)
        else:
            return None

    def P_dis(self, data_frame):
        '''
        Calculate the probabilities based of match outcomes.

        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'

        Returns:
        pandas.DataFrame: 'DataFrame' indexed by `'MatchID'`s and the associated outcome probabilities `'P(H)'`, `'P(D)'` and `'P(A)'`.
        '''
        P_dis = pd.DataFrame(columns=['P(H)', 'P(D)', 'P(A)'])

        for (MatchID, HID, AID) in data_frame[['MatchID','HID','AID']].values:
            P_dis = P_dis.append(self.P_dis_match(MatchID,HID,AID))

        return P_dis

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
