import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GS_season:
    def __init__(self):
        #frame of teams
        self.teams = pd.DataFrame(columns=["GS"]) 
        
    def eval_opps(self, opps):
        '''
        Evaluate betting opportunities:
            Adds previously unknown teams to `self.teams`
        '''
        self.eval_new_teams(opps)    

    def eval_inc(self, inc):
        '''
        Evaluate data increment:
            1) Adds previously unknown teams to `self.teams`
            2) Evaluates the new goals scored for the teams
        '''
        self.eval_new_teams(inc)
        self.eval_update_GS(inc)
    
    def eval_new_teams(self, data_frame):
        '''
        New teams in `data_frame` to `self.teams` and associate goals scored 0 with them. (appends to `self.teams`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        # FIXME: This could be done in one run through cating the 'HID' and the 'AID' cols <09-11-20, kunzaatko> #
        new_teams = data_frame[[home_team not in self.teams.index for home_team in data_frame['HID']]]['HID'].append(data_frame[[ away_team not in self.teams.index for away_team in data_frame['AID']]]['AID']).unique()

        for team in new_teams:
            self.teams = self.teams.append(pd.DataFrame(data={'GS': [0]},index=[team]))
    
    
    def run_iter(self, inc, opps):
        '''
        Run the iteration of the evaluation loop.
        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                Has to include 'Sea', HID', 'AID', 'HSC' and 'ASC'.
        opps(pandas.DataFrame): 'DataFrame' with the betting opportunities.
                                Has to include 'MatchID', 'HID' and 'AID'.
        Returns:
        pandas.DataFrame or None: 'DataFrame' indexed by `'MatchID'`s and the a  `'H-GS'` and `'A-GS'` for all matches in `opps` or `None` if no `opps` where passed.
        !WARNING GS in first match in season is complete GS from last season!
        '''
        if inc is not None:
            self.eval_inc(inc)

        if opps is not None:
            self.eval_opps(opps)
            return self.GS(opps)
        else:
            return None
 
         
            
    def eval_update_GS(self, data_frame):
        '''
        Updates the GS for based on the games recorded in the `data_frame`. (mutates `self.teams['GS']`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                        Has to include 'Sea' 'HID', 'AID', 'HSC' and 'ASC'.
        '''
        
        cSea = data_frame['Sea'].min() #fake Sea to help change GS
        for (Sea, HID, AID, HSC, ASC) in data_frame[['Sea', 'HID','AID','HSC','ASC']].values:
            ''' compare Sea cSea, if they are equal add goal scored
                else reset GS and update cSea'''
            if Sea == cSea:
                self.teams.at[HID, 'GS'] += HSC
                self.teams.at[AID, 'GS'] += ASC
            else:
                self.teams.at[HID, 'GS'] = HSC
                self.teams.at[AID, 'GS'] = ASC
                cSea = Sea
                
        
    def GS(self, data_frame):
        '''
        Calculate the probabilities based of match outcomes.
        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'
        Returns:
        pandas.DataFrame: 'DataFrame' indexed by `'MatchID'`s and the associated scored goals of teams in season `'H-GS'`, `'A-GS'`.
        '''
        GS = pd.DataFrame(columns=['H-GS', 'A-GS'])

        for (MatchID, HID, AID) in data_frame[['MatchID','HID','AID']].values:
             GS = GS.append(self.GS_match(MatchID,HID,AID))

        return GS

    def GS_match(self, MatchID, HID, AID):
        '''
        Calculate goal scored in last match.
        Parameters:
        MatchID(int): The ID of the match. From the column 'MatchID' of the `data_frame`.
        HID(int): The ID of the home team. From the column 'HID' of the `data_frame`.
        AID(int): The ID of the home team. From the column 'AID' of the `data_frame`.
        Returns:
        pandas.DataFrame: 'DataFrame' with one row with the index `'MatchID'` and the associated scored goals of teams in season `'H-GS'` and `'A-GS'`.
        '''
        
        
        [home_GS, away_GS] = [self.teams.at[ID,'GS'] for ID in [HID,AID]]

    
        return pd.DataFrame(data={'H-GS': [home_GS], 'A-GS': [away_GS]}, index=[MatchID])
