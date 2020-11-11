Python 3.6.8 (default, Mar 21 2019, 10:08:12) 
[GCC 8.3.1 20190223 (Red Hat 8.3.1-2)] on linux
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Goals:
    def __init__(self):
        #frame of teams
        self.teams = pd.DataFrame(columns=["Goals"]) 
    
    def eval_new_teams(self, data_frame):
        '''
        New teams in `data_frame` to `self.teams` and associate goals 0 with them. (appends to `self.teams`)
        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        # FIXME: This could be done in one run through cating the 'HID' and the 'AID' cols <09-11-20, kunzaatko> #
        new_teams = data_frame[[home_team not in self.teams.index for home_team in data_frame['HID']]]['HID'].append(data_frame[[ away_team not in self.teams.index for away_team in data_frame['AID']]]['AID']).unique()

        for team in new_teams:
            self.teams = self.teams.append(pd.DataFrame(data={'Goals': [0]},index=[team]))
    
        
    def Goals(self, data_frame):
        '''
        Calculate the probabilities based of match outcomes.
        Parameters:
        data_frame(pandas.DataFrame):   `data_frame` with matches. (`opps`).
                                        Has to include 'MatchID', 'HID' and 'AID'
        Returns:
        pandas.DataFrame: 'DataFrame' indexed by `'MatchID'`s and the associated scored goals of teams in season `'H-GS'`, `'A-GS'`.
        '''
        Goals = pd.DataFrame(columns=['H-Goals', 'A-Goals'])

        for (MatchID, HID, AID) in data_frame[['MatchID','HID','AID']].values:
             Goals = Goals.append(self.Goals_match(MatchID,HID,AID))

        return Goals

    def Goals_match(self, MatchID, HID, AID):
        '''
        Calculate goal scored in last match.
        Parameters:
        MatchID(int): The ID of the match. From the column 'MatchID' of the `data_frame`.
        HID(int): The ID of the home team. From the column 'HID' of the `data_frame`.
        AID(int): The ID of the home team. From the column 'AID' of the `data_frame`.
        Returns:
        pandas.DataFrame: 'DataFrame' with one row with the index `'MatchID'` and the associated scored goals of teams in season `'H-GS'` and `'A-GS'`.
        '''
        
        
        [home_GS, away_GS] = [self.teams.at[ID,'Goals'] for ID in [HID,AID]]

    
        return pd.DataFrame(data={'H-Goals': [home_GS], 'A-Goals': [away_GS]}, index=[MatchID])
    
    
class WrongNameError(Exception):
    """Exception raised for errors in the input Goals characterics.

    Attributes:
        characteristcis -- input name which caused the error
        message -- explanation of the error
    """

    def __init__(self,name, message="This characteristics doesn't exit, try 'GS_per_Season', 'GC_per_Season' or G_Difference_Season'"):
        self.characterics = name
        self.message = message
        super().__init__(self.message)


    
    
class Goals_scored_season(Goals):
        def __init__(self):
            Goals.__init__(self)
            '''
            Goals scored in season
            '''
        
        def eval_update_GS(self, data_frame):
            '''
            Updates the Goals for based on the games recorded in the `data_frame`. (mutates `self.teams['Goals']`)
            Parameters:
            data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                            Has to include 'Sea' 'HID', 'AID', 'HSC' and 'ASC'.
            '''
            cSea = data_frame['Sea'].min() #fake Sea to help change GS
            for (Sea, HID, AID, HSC, ASC) in data_frame[['Sea', 'HID','AID','HSC','ASC']].values:
                ''' compare Sea cSea, if they are equal add goal scored
                    else reset GS and update cSea'''
                if Sea == cSea:
                    self.teams.at[HID, 'Goals'] += HSC
                    self.teams.at[AID, 'Goals'] += ASC
                else:
                    self.teams.at[HID, 'Goals'] = HSC
                    self.teams.at[AID, 'Goals'] = ASC
                    cSea = Sea

class Goals_conceded_season(Goals):
        def __init__(self):
            Goals.__init__(self)
            '''
            Goals conceded in season
            '''
            
        def eval_update_GC(self, data_frame):
            '''
            Updates the Goals for based on the games recorded in the `data_frame`. (mutates `self.teams['Goals']`)
            Parameters:
            data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                            Has to include 'Sea' 'HID', 'AID', 'HSC' and 'ASC'.
            '''

            cSea = data_frame['Sea'].min() #fake Sea to help change GC
            for (Sea, HID, AID, HSC, ASC) in data_frame[['Sea', 'HID','AID','HSC','ASC']].values:
                ''' 
                    compare Sea cSea, if they are equal add goal scored
                    else reset Goals and update cSea
                '''
                if Sea == cSea:
                    self.teams.at[HID, 'Goals'] += ASC
                    self.teams.at[AID, 'Goals'] += HSC
                else:
                    self.teams.at[HID, 'Goals'] = ASC
                    self.teams.at[AID, 'Goals'] = HSC
                    cSea = Sea

                
class Goals_Different_season(Goals):
        def __init__(self):
            Goals.__init__(self)
            '''
            Difference between scored and conceded goals in season
            '''
            
        def eval_update_G_Difference(self, data_frame):
            '''
            Updates the Goals for based on the games recorded in the `data_frame`. (mutates `self.teams['Goals']`)
            Parameters:
            data_frame (pandas.DataFrame):  `data_frame` where the games played are recorded.
                                            Has to include 'Sea' 'HID', 'AID', 'HSC' and 'ASC'.
            '''

            cSea = data_frame['Sea'].min() #fake Sea to help change GC
            for (Sea, HID, AID, HSC, ASC) in data_frame[['Sea', 'HID','AID','HSC','ASC']].values:
                ''' 
                    compare Sea cSea, if they are equal add G_Different
                    else reset G_Different and update cSea
                '''
                if Sea == cSea:
                    self.teams.at[HID, 'Goals'] += (HSC-ASC)
                    self.teams.at[AID, 'Goals'] += (ASC-HSC)
                else:
                    self.teams.at[HID, 'Goals'] = (HSC-ASC)
                    self.teams.at[AID, 'Goals'] = (ASC-HSC)
                    cSea = Sea
                    
class Goals_scored_last(Goals):
        def __init__(self):
            Goals.__init__(self)
            
        def eval_update_GSlast(self, data_frame):
            '''
             Goals scored in last match
            '''
            for (HID, AID, HSC, ASC) in data_frame[['HID','AID','HSC','ASC']].values:
                self.teams.at[HID, 'Goals'] = HSC
                self.teams.at[AID, 'Goals'] = ASC

class Goals_conceded_last(Goals):
        def __init__(self):
            Goals.__init__(self)
            
        def eval_update_GClast(self, data_frame):
            '''
            Goals conceded in last match
            '''
            for (HID, AID, HSC, ASC) in data_frame[['HID','AID','HSC','ASC']].values:
                self.update_GSlast(HID, AID, HSC, ASC)
                self.teams.at[HID, 'Goals'] = ASC
                self.teams.at[AID, 'Goals'] = HSC

                    
class Goals_Characteristics(Goals_scored_season, Goals_conceded_season, Goals_Different_season, Goals_scored_last, Goals_conceded_last):
        def __init__(self, name):
            ''' 
                Called class for Goals characteristics
                Atributes : name of characteristics
                chosable characteristics: 'GS_per_Season', 'GC_per_Season', 'G_Difference_Season, 'GS_last' or 'GC_last'
            '''
            Goals_scored_season.__init__(self)
            Goals_conceded_season.__init__(self)
            Goals_Different_season.__init__(self)
            Goals_scored_last.__init__(self)
            Goals_conceded_last.__init__(self)
            if not (name == 'GS_per_Season' or name == 'GC_per_Season' or name == 'G_Difference_Season' 
                    or name == 'GS_last' or name == 'GC_last'):
                print(name)
                raise WrongNameError(name)
            self.name = name
       
        def eval_opps(self, opps):
            '''
            Evaluate betting goals characteristics:
                Adds previously unknown teams to `self.teams`
            '''
            self.eval_new_teams(opps)    

        def eval_inc(self, inc):
            '''
            Evaluate data increment with respect to Goals characteristics:
                1) Adds previously unknown teams to `self.teams`
                2) Evaluates the new goals scored for the teams
            '''
            self.eval_new_teams(inc)
            if self.name == 'GS_per_Season':
                self.eval_update_GS(inc)
            elif self.name == 'GC_per_Season':
                self.eval_update_GC(inc)
            elif self.name == 'G_Difference_Season':
                self.eval_update_G_Difference(inc)
            elif self.name == 'GS_last':
                self.eval_update_GSlast(inc)
            elif self.name == 'GC_last':
                self.eval_update_GClast(inc)

        def run_iter(self, inc, opps):
            '''
            Run the iteration of the evaluation loop.
            Parameters:
            inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                    Has to include 'Sea', HID', 'AID', 'HSC' and 'ASC'.
            opps(pandas.DataFrame): 'DataFrame' with the betting opportunities.
                                    Has to include 'MatchID', 'HID' and 'AID'.
            Returns:
            pandas.DataFrame or None: 'DataFrame' indexed by `'MatchID'`s and the a  `'H-Goals'` and `'A-Goals'` of chosen goal characteristics for all matches in `opps` or `None` if no `opps` where passed.
            !WARNING in season period goals characterics the goals in first match in season are complete goals from the last season !
            '''
            if inc is not None:
                self.eval_inc(inc)

            if opps is not None:
                self.eval_opps(opps)
                return self.Goals(opps)
            else:
                return None
