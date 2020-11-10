import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class G_Different_season:
    def __init__(self):
        #frame of teams
        self.teams = pd.DataFrame(columns=["G_Different"]) 
        
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
        self.eval_update_G_Different(inc)
    
    def eval_new_teams(self, data_frame):
        '''
        New teams in `data_frame` to `self.teams` and associate G-Different scored 0 with them. (appends to `self.teams`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        # FIXME: This could be done in one run through cating the 'HID' and the 'AID' cols <09-11-20, kunzaatko> #
        new_teams = data_frame[[home_team not in self.teams.index for home_team in data_frame['HID']]]['HID'].append(data_frame[[ away_team not in self.teams.index for away_team in data_frame['AID']]]['AID']).unique()

        for team in new_teams:
            self.teams = self.teams.append(pd.DataFrame(data={'G_Different': [0]},index=[team]))
    
    
    def run_iter(self, inc, opps):
        '''
        Run the iteration of the evaluation loop.
        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                Has to include 'Sea', HID', 'AID', 'HSC' and 'ASC'.
        opps(pandas.DataFrame): 'DataFrame' with the betting opportunities.
                                Has to include 'MatchID', 'HID' and 'AID'.
        Returns:
        pandas.DataFrame or None: 'DataFrame' indexed by `'MatchID'`s and the a  `'H-G_Different'` and `'A-G_Different'` for all matches in `opps` or `None` if no `opps` where passed.
        !WARNING G_Different in first match in season is complete G_Different from last season!
        '''
        if inc is not None:
            self.eval_inc(inc)

        if opps is not None:
            self.eval_opps(opps)
            return self.G_Different(opps)
        else:
            return None
 
         
            
    def eval_update_G_Different(self, data_frame):
        '''
        Updates the G_Different for based on the games recorded in the `data_frame`. (mutates `self.teams['G_Different']`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                        Has to include 'Sea' 'HID', 'AID', 'HSC' and 'ASC'.
        '''
        
        cSea = data_frame['Sea'].min() #fake Sea to help change GC
        for (Sea, HID, AID, HSC, ASC) in data_frame[['Sea', 'HID','AID','HSC','ASC']].values:
            ''' compare Sea cSea, if they are equal add G_Different
                else reset G_Different and update cSea'''
            if Sea == cSea:
                self.teams.at[HID, 'G_Different'] += (HSC-ASC)
                self.teams.at[AID, 'G_Different'] += (ASC-HSC)
            else:
                self.teams.at[HID, 'G_Different'] = (HSC-ASC)
                self.teams.at[AID, 'G_Different'] = (ASC-HSC)
                cSea = Sea
                
        
    def G_Different(self, data_frame):
        '''
        Calculate G-Differrent based on last matches in season.
        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'
        Returns:
        pandas.DataFrame: 'DataFrame' indexed by `'MatchID'`s and the associated G_Different of teams in season `'H-G_Different'`, `'A-G_Different'`.
        '''
        G_Different = pd.DataFrame(columns=['H-G_Different', 'A-G_Different'])

        for (MatchID, HID, AID) in data_frame[['MatchID','HID','AID']].values:
             G_Different = G_Different.append(self.G_Different_match(MatchID,HID,AID))

        return G_Different

    def G_Different_match(self, MatchID, HID, AID):
        '''
        Calculate G-Different in last matches in season.
        Parameters:
        MatchID(int): The ID of the match. From the column 'MatchID' of the `data_frame`.
        HID(int): The ID of the home team. From the column 'HID' of the `data_frame`.
        AID(int): The ID of the home team. From the column 'AID' of the `data_frame`.
        Returns:
        pandas.DataFrame: 'DataFrame' with one row with the index `'MatchID'` and the G_Different of teams in season `'H-G_Different'` and `'A-G_Different'`.
        '''
        
        
        [home_G_Different, away_G_Different] = [self.teams.at[ID,'G_Different'] for ID in [HID,AID]]

    
        return pd.DataFrame(data={'H-G_Different': [home_G_Different], 'A-G_Different': [away_G_Different]}, index=[MatchID])
