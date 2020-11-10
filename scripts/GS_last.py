import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GS_last:
    def __init__(self):
        #frame of teams
        self.teams = pd.DataFrame(columns=["GS_last"]) 
            
    def eval_opps(self, opps):
        '''
        Evaluate lastGS:
             Adds previously unknown teams to `self.teams`
        '''
        self.eval_new_teams(opps)

    def eval_inc(self, inc):
        '''
        Evaluate data increment:
            1) Adds previously unknown teams to `self.teams`
            2) Evaluates the new last GS for the teams
        '''
        self.eval_new_teams(inc)
        self.eval_update_GSlast(inc)
        
    
    def eval_new_teams(self, data_frame):
        '''
        New teams in `data_frame` to `self.teams` and associate 0 GS in last match with them. (appends to `self.teams`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        # FIXME: This could be done in one run through cating the 'HID' and the 'AID' cols <09-11-20, kunzaatko> #
        new_teams = data_frame[[home_team not in self.teams.index for home_team in data_frame['HID']]]['HID'].append(data_frame[[ away_team not in self.teams.index for away_team in data_frame['AID']]]['AID']).unique()

        for team in new_teams:
            self.teams = self.teams.append(pd.DataFrame(data={'GSlast': [0]},index=[team]))
         
            
    def eval_update_GSlast(self, data_frame):
        '''
        Updates the GSlast for based on the games recorded in the `data_frame`. (mutates `self.teams['GS_last']`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                        Has to include 'HID', 'AID', 'HSC', 'ASC'.
        '''
        for (HID, AID, HSC, ASC) in data_frame[['HID','AID','HSC','ASC']].values:
            self.update_GSlast(HID, AID, HSC, ASC)
    
    
    def run_iter(self, inc, opps):
        '''
        Run the iteration of the evaluation loop.
        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                Has to include 'HID', 'AID', 'LID', 'H', 'D' and 'A'.
        opps(pandas.DataFrame): 'DataFrame' with the betting opportunities.
                                Has to include 'MatchID', 'HID' and 'AID'.
        Returns:
        pandas.DataFrame or None: 'DataFrame' indexed by `'MatchID'`s and the associated outcome probabilities `'H-GSlast'` and `'A-GSlast'` for all matches in `opps` or `None` if no `opps` where passed.
        '''
        if inc is not None:
            self.eval_inc(inc)

        if opps is not None:
            self.eval_opps(opps)
            return self.GSlast(opps)
        else:
            return None
    

    def update_GSlast(self, HID, AID, HSC, ASC):
        '''
         one match goal score
        '''
        self.teams.at[HID, 'GS_last'] = HSC
        self.teams.at[AID, 'GS_last'] = ASC
        
    def GSlast(self, data_frame):
        '''
        Calculate the probabilities based of match outcomes.
        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'
        Returns:
        pandas.DataFrame: 'DataFrame' indexed by `'MatchID'`s and the associated outcome probabilities `'H-GSlast'`, `'A-GSlast'`.
        '''
        GSlast = pd.DataFrame(columns=['H-GSlast', 'A-GSlast'])

        for (MatchID, HID, AID) in data_frame[['MatchID','HID','AID']].values:
             GSlast = GSlast.append(self.GSlast_match(MatchID,HID,AID))

        return GSlast

    def GSlast_match(self, MatchID, HID, AID):
        '''
        Calculate goal scored in last match.
        Parameters:
        MatchID(int): The ID of the match. From the column 'MatchID' of the `data_frame`.
        HID(int): The ID of the home team. From the column 'HID' of the `data_frame`.
        AID(int): The ID of the home team. From the column 'AID' of the `data_frame`.
        Returns:
        pandas.DataFrame: 'DataFrame' with one row with the index `'MatchID'` and the associated outcome probabilities `'P(H)'`, `'P(D)'` and `'P(A)'`.
        '''

        [home_GSlast, away_GSlast] = [self.teams.at[ID,'GS_last'] for ID in [HID,AID]]

    
        return pd.DataFrame(data={'H-GSlast': [home_GSlast], 'A-GSlast': [away_GSlast]}, index=[MatchID])
