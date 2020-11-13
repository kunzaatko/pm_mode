import numpy as np
import pandas as pd

class Data:
    def __init__(self):
        '''
        Class for manipulating the data and extracting characteristics.

        enum of (wanted) characteristics:
        ---------------------------------

        ##########################
        #  TIME CHARACTERISTICS  #
        ##########################
        Season-long: season-long score ('SL_SC'), season-long win-lose ratio ('SL_RES'), season-long matches played ('SL_PLAY')
        Life-long: life-long score ('LL_SC'), life-long win-lose ratio ('LL_RES'), life-long matches played ('LL_PLAY')

        ###########################
        #  MATCH CHARACTERISTICS  #
        ###########################
        Last-match: last-match score ('LM_SC'), last-match result ('LM_RES'), last-match oppo ('LM_OPPO'), last-match P_dis ('LM_P_DIS') last-match date ('LM_DATE')
        Last-match-with: last-match-with score ('LMW_SC'), last-match-with result ('LMW_RES'), last-match-with P_dis ('LMW_P_DIS'), last-match-with date ('LMW_DATE')
        '''

        ##################
        #  Storage data  #
        ##################

        self.today = None # current date
        self.curr_opps = None # current `opps`
        self.curr_inc = None # current `inc`
        self.betting_runs_log = None # `opps` that was passed with the associated `P_dis`, and the associated bets indexed by date # NOTE: This potentialy could slow down the system and occupy more RAM <13-11-20, kunzaatko> #
        self.last_summary = None # last `summary` that was passed
        self.matches = None # All matches played by IDs


        ##############################
        #  Features data - by teams  #
        ##############################

        # 'LID = league ID'
        self.team_index = pd.DataFrame(columns=['LID']) # recorded teams

        # 'SC = score', 'RES = result', 'PLAY = matches played', 'NEW = new', 'ACU = accuracy'
        self.time_data = pd.DataFrame(columns=['SL_SC', 'SL_RES', 'SL_PLAY', 'SL_NEW', 'SL_ACCU', 'LL_SC', 'LL_RES', 'LL_PLAY', 'LL_NEW', 'LL_ACCU']) # data frame for storing all the time characteristics
        self.season_time_data = pd.DataFrame(columns=['S_DATA_FRAME']) # data frame for moving the data of the last season when a new season in started

        # 'SC = score', 'RES = result', 'DATE = date', 'LM_SIDE = home/away', 'LM_P_DIS = [win_p, draw_p, lose_p]'
        self.last_match_data = pd.DataFrame(columns=['MatchID', 'LM_SC', 'LM_RES', 'LM_DATE', 'LM_SIDE']) # data frame for storing all the Last-match characteristics
        self.matches_data = pd.DataFrame(columns=['M_DATA_FRAME']) # data frame for moving the data of the last match when a new match is played


    def update(self, opps ,summary, inc, P_dis):
        '''
        Run the iteration update of the characteristics and features.

        Parameters:
        All the parameters are supplied by the evaluation loop.
        opps(pandas.DataFrame): dataframe that includes the opportunities for betting.
        summary(pandas.DataFrame): includes the `Max_bet`, `Min_bet` and `Bankroll`.
        inc(pandas.DataFrame): includes the played matches with the scores for the model.
        '''
        self.__eval_summary(summary)
        self.__eval_opps(opps)
        self.__eval_inc(inc)
        self.__eval_P_dis(P_dis)

    def __eval_summary(self, summary):
        '''
        Evaluate the `summary` dataframe.

        Parameters:
        summary(pandas.DataFrame):  Summary of the current iteration. (from env)
                                        Has to include 'Bankroll', 'Max_bet' and 'Min_bet'.
        '''
        pass

    def __eval_opps(self, opps):
        '''
        Evaluate the `opps` dataframe.

        Parameters:
        opps(pandas.DataFrame): `DataFrame` that includes the opportunities for betting.
        '''
        self.__eval_new_teams(opps)
        self.__eval_new_LIDs(opps)
        self.last_opps = opps
        # TODO:  <13-11-20, kunzaatko> #

    def __eval_inc(self,inc):
        '''
        Evaluate the `inc` dataframe.

        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
        '''
        self.__eval_new_teams(inc)
        self.__eval_new_LIDs(inc)

    # TODO: Make this method run faster. It will take a life time to be run if we want to run it as offten as we do. <13-11-20, kunzaatko> #
    def __eval_new_teams(self, data_frame):
        '''
        Add new teams to `self.team_index`.

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        new_home_teams = data_frame[[home_team not in self.team_index.index for home_team in data_frame['HID'].values]] # new teams from the 'HID' column of `data_frame`
        new_away_teams = data_frame[[ away_team not in self.team_index.index for away_team in data_frame['AID'].values]] # new teams from the 'AID' column of `data_frame`
        new_teams = pd.DataFrame()

        if not new_home_teams.empty:
            new_teams = new_teams.append(new_home_teams['HID'])
        if not new_away_teams.empty:
            new_teams = new_teams.append(new_away_teams['AID'])

        for team in new_teams:
            self.team_index = self.team_index.append(pd.DataFrame(data={'LID': [[]]}, index=[team]))

    # TODO: Make this method run faster. It will take a life time to be run if we want to run it as offten as we do. <13-11-20, kunzaatko> #
    def __eval_new_LIDs(self, data_frame):
        '''
        If team is playing in a league that it did not play before, associate the 'LID' with it. (mutates `self.teams['LIDs']`)

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new LIDs from for the teams in `self.teams` (`inc` and `opps`).
                                        Has to include 'HID', 'AID' and 'LID'.
        '''
        for team in self.team_index.index:
            # TODO: use pandas dataframe for this <10-11-20, kunzaatko> #
            LIDs = []
            if team in data_frame['HID'].values:
                for LID in data_frame.set_index('HID').at[team,'LID']:
                    if LID not in LIDs:
                        LIDs.append(LID)
            elif team in data_frame['AID'].values:
                for LID in data_frame.set_index('AID').at[team,'LID']:
                    if LID not in LIDs:
                        LIDs.append(LID)

            for LID in LIDs:
                if LID not in self.team_index.at[team,'LID']:
                    self.team_index.at[team, 'LID'].append(LID)



    def __eval_P_dis(self, data_frame):
        '''
        Associate the P_dis with the team.
        '''


    # ┌─────────────────┐
    # │ LAST MATCH WITH │
    # └─────────────────┘

    def last_match_with(self, oppo_ID, ID=None):
        '''
        Characteristics of the last match with specific opponent.

        Parameters:
        oppo_ID(int): ID of the opponent.
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    def last_match_with_score(self, oppo_ID, ID=None):
        '''
        Score in the last match with specific opponent.

        Parameters:
        oppo_ID(int): ID of the opponent.
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    def last_match_with_date(self, oppo_ID, ID=None):
        '''
        Returns the date of the last match with specific opponent.

        Parameters:
        oppo_ID(int): ID of the opponent.
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass


    # ┌────────────┐
    # │ LAST MATCH │
    # └────────────┘

    def last_match(self, ID=None):
        '''
        Characteristics of the last match.

        Parameters:
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    def last_match_score(self, ID=None):
        '''
        Score last match.

        Parameters:
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    def last_match_date(self, ID=None):
        '''
        Returns the date of the last match.

        Parameters:
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    # ┌─────────────────────┐
    # │ MATCHES GLOBAL DATA │
    # └─────────────────────┘

    def matches_with(self, oppo_ID, ID=None):
        '''
        Returns all the matches with a particular opponent.

        Parameters:
        oppo_ID(int): ID of the opponent.
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    # ┌───────────────────────────┐
    # │ LIFE-LONG CHARACTERISTICS │
    # └───────────────────────────┘

    def total_score_to_match(self, ID=None):
        '''
        Total life-long score to match ratio. ()

        Parameters:
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

    def home_to_away_r(self, ID=None):
        '''
        Win rate for home and away.

        Parameters:
        ID(int/None): If `None`, then it is returned for all teams. If `ID=id` then only returned for the team `id`.
        '''
        pass

