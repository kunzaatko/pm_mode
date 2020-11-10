import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GC_last:
    def __init__(self):
        #frame of teams
        self.teams = pd.DataFrame(columns=["GC_last"]) 
    
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
            2) Evaluates the new last GC for the teams
        '''
        self.eval_new_teams(inc)
        self.eval_update_GClast(inc)
    
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
            self.teams = self.teams.append(pd.DataFrame(data={'GClast': [0]},index=[team]))

            
    def eval_update_GClast(self, data_frame):
        '''
        Updates the GClast for based on the games recorded in the `data_frame`. (mutates `self.teams['GC_last']`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                        Has to include 'HID', 'AID', 'HSC' and 'ASC'.
        '''
        for (HID, AID, HSC, ASC) in data_frame[['HID','AID','HSC','ASC']].values:
            self.update_GClast(HID, AID, HSC, ASC)
    
    def run_iter(self, inc, opps):
        '''
        Run the iteration of the evaluation loop.
        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                Has to include 'HID', 'AID', 'LID', 'HSC' and 'ASC'.
        opps(pandas.DataFrame): 'DataFrame' with the betting pportunities.
                                Has to include 'MatchID', 'HID' and 'AID'.
        Returns:
        pandas.DataFrame or None: 'DataFrame' indexed by `'MatchID'`s and the associated outcome probabilities `'H-GClast'` and `'A-GClast'` for all matches in `opps` or `None` if no `opps` where passed.
        '''
        if inc is not None:
            self.eval_inc(inc)

        if opps is not None:
            self.eval_opps(opps)
            return self.GClast(opps)
        else:
            return None
    
    

    def update_GClast(self, HID, AID, HSC, ASC):
        '''
        update one match goal score
        '''
        self.teams.at[HID, 'GC_last'] = ASC
        self.teams.at[AID, 'GC_last'] = HSC
        
    def GClast(self, data_frame):
        '''
        Calculate the probabilities based of match outcomes.
        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'
        Returns:
        pandas.DataFrame: 'DataFrame' indexed by `'MatchID'`s and the associated GC last `'H-GClast'` and `'A-GClast'`.
        '''
        GClast = pd.DataFrame(columns=['H-GClast', 'A-GClast'])

        for (MatchID, HID, AID) in data_frame[['MatchID','HID','AID']].values:
             GClast = GClast.append(self.GClast_match(MatchID,HID,AID))

        return GClast

    def GClast_match(self, MatchID, HID, AID):
        '''
        Calculate goal  in last match.
        Parameters:
        MatchID(int): The ID of the match. From the column 'MatchID' of the `data_frame`.
        HID(int): The ID of the home team. From the column 'HID' of the `data_frame`.
        AID(int): The ID of the home team. From the column 'AID' of the `data_frame`.
        Returns:
        pandas.DataFrame: 'DataFrame' with one row with the index `'MatchID'` and the associated GC last `'H-GClast'` and `'A-GClast'`.
        '''

        [home_GClast, away_GClast] = [self.teams.at[ID,'GC_last'] for ID in [HID,AID]]

    
        return pd.DataFrame(data={'H-GClast': [home_GClast], 'A-GClast': [away_GClast]}, index=[MatchID])
