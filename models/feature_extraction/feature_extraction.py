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

        self.today = None # current date ﭾ
        self.curr_opps = None # current `opps` ﭾ
        self.curr_inc = None # current `inc` ﭾ
        self.curr_summary = None # current `summary` ﭾ
        self.curr_bets = None # current `bets` ﭾ
        self.curr_teams = None # teams in the current `inc`/`opps`
        self.curr_P_dis = None # current `P_dis` ﭾ
        self.curr_betting_run = None # current `opps` associated with `P_dis`, and the associated `bets`ﭾ
        self.betting_runs = {} # `opps` that was passed with the associated `P_dis`, and the associated `bets` (dict) # NOTE: This potentialy could slow down the system and occupy more RAM <13-11-20, kunzaatko> #
        self.matches = pd.DataFrame(columns=['Date', 'Sea','LID','HID','AID','OddsH','OddsD','OddsA','HSC','ASC','H','D','A','BetH','BetD','BetA']) # All matches played by IDs ﭾ
        # 'LID = league ID (list)'
        self.team_index = pd.DataFrame(columns=['LID']) # recorded teams ﭾ


        ##############################
        #  Features data - by teams  #
        ##############################

        # 'SC = score (pd.DataFrame(columns=['TEAM', 'OPPO']))', 'RES = result (pd.DataFrame(columns=['TEAM', 'DRAW', 'OPPO']))', 'PLAYED = matches played (int)', 'NEW = new (bool)', 'ACU = accuracy (float)'
        self.time_data = pd.DataFrame(columns=['SL_SC', 'SL_RES', 'SL_PLAYED', 'SL_NEW', 'SL_ACCU', 'LL_SC', 'LL_RES', 'LL_PLAYED', 'LL_NEW', 'LL_ACCU']) # data frame for storing all the time characteristics
        self.season_time_data = pd.DataFrame(columns=['S_DATA_FRAME']) # data frame for moving the data of the last season when a new season in started

        # 'SC = score (TEAM, OPPO)', 'RES = result (pd.DataFrame(columns=['TEAM', 'DRAW', 'OPPO']))', 'DATE = date', 'LM_SIDE = home/away (str)', 'LM_P_DIS = pd.DataFrame(columns=['win_p', 'draw_p', 'lose_p'])'
        self.last_match_data = pd.DataFrame(columns=['MatchID', 'LM_SC (T,O)', 'LM_RES (T,D,O)', 'LM_DATE', 'LM_SIDE (H,A)', 'LM_P_DIS (W,D,L)']) # data frame for storing all the Last-match characteristics
        self.matches_data = pd.DataFrame(columns=['M_DATA_FRAME']) # data frame for moving the data of the last match when a new match is played


    ######################################
    #  UPDATING THE DATA STORED IN SELF  #
    ######################################

    def update_data(self, opps=None ,summary=None, inc=None, P_dis=None, bets=None):
        '''# {{{
        Run the iteration update of the data stored.

        Parameters:
        All the parameters are supplied by the evaluation loop.
        opps(pandas.DataFrame): dataframe that includes the opportunities for betting.
        summary(pandas.DataFrame): includes the `Max_bet`, `Min_bet` and `Bankroll`.
        inc(pandas.DataFrame): includes the played matches with the scores for the model.
        '''
        if summary is not None:
            self.__eval_summary(summary)

        if inc is not None:
            self.__eval_inc(inc)

        if opps is not None:
            self.__eval_opps(opps)

        if P_dis is not None:
            self.__eval_P_dis(P_dis)

        if bets is not None:
            self.__eval_bets(bets)

        # If all the data needed was already recorded put it into the dictionary.
        if self.curr_opps is not None and self.curr_summary is not None and self.curr_bets is not None and self.curr_P_dis is not None:
            self.betting_runs[self.today] = self.curr_betting_run # }}}

    def __eval_opps(self, opps):
        '''# {{{
        Evaluate the `opps` dataframe.

        Parameters:
        opps(pandas.DataFrame): `DataFrame` that includes the opportunities for betting.
        '''
        self.__eval_new_teams(opps)
        self.__eval_new_LIDs(opps)
        self.__eval_curr_betting_run(opps)
        self.__eval_matches(opps)
        self.__eval_curr_betting_run(opps)
        self.curr_opps = opps # }}}

    def __eval_inc(self,inc):
        ''' # {{{
        Evaluate the `inc` dataframe.

        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
        '''
        self.curr_inc = inc # }}}
        self.__eval_new_teams(inc)
        self.__eval_new_LIDs(inc)
        self.__eval_matches(inc)

    def __eval_summary(self, summary):
        ''' # {{{
        Evaluate the `summary` dataframe.

        Parameters:
        summary(pandas.DataFrame):  Summary of the current iteration. (from env)
                                        Includes 'Bankroll', 'Max_bet' and 'Min_bet'.
        '''
        self.today = summary['Date'][0]
        self.curr_summary = summary # }}}

    def __eval_bets(self,bets):
        '''# {{{
        Evaluate the `bets` dataframe.

        Parameters:
        bets(pandas.DataFrame): cast bets idexed by 'MatchID'
        '''
        self.__eval_matches(bets, check_for_new=False)
        self.__eval_curr_betting_run(bets)
        self.curr_bets = bets# }}}

    def __eval_P_dis(self, P_dis):
        ''' # {{{
        Associate the P_dis with the match.
        '''
        self.__eval_curr_betting_run(P_dis)
        self.curr_P_dis = P_dis# }}}

    # TODO: Make this method run faster. It will take a life time to be run if we want to run it as offten as we do. <13-11-20, kunzaatko> #
    def __eval_new_teams(self, data_frame):
# {{{
        '''
        Add new teams to `self.team_index`.

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new teams from (`inc` and `opps`).
                                        Has to include 'HID' and 'AID'.
        '''
        if not data_frame.empty:
            new_teams = pd.concat((data_frame['HID'], data_frame['AID'])).unique()
            new_teams = new_teams[[new for new in range(len(new_teams)) if new_teams[new] not in self.team_index.index]]
        else:
            new_teams = []

        for team in new_teams:
            if team not in self.team_index.index:
                self.team_index.loc[team] = {'LID': []}
# }}}

    def __eval_new_LIDs(self, data_frame):
        ''' # {{{
        If team is playing in a league that it did not play before, associate the 'LID' with it. (mutates `self.teams['LIDs']`)

        Parameters:
        data_frame (pandas.DataFrame):  `data_frame` to get new LIDs from for the teams in `self.teams` (`inc` and `opps`).
                                        Has to include 'HID', 'AID' and 'LID'.
        '''

        ID_LID = np.concatenate((data_frame[['HID', 'LID']].values,data_frame[['AID','LID']].values))

        for ID,LID in ID_LID:
            if LID not in self.team_index.at[ID,'LID']:
                self.team_index.at[ID,'LID'].append(LID)# }}}

    def __eval_curr_teams(self, data_frame):
        '''# {{{
        Evaluate current teams from the data frame. They are used for better iterating and updating the features.
        '''
        self.curr_teams = np.concatenate((data_frame['HID'].values, data_frame['AID'].values))# }}}

    def __eval_curr_betting_run(self, data_frame):
        ''' # {{{
        Evaluate the data_frame and add the data to the current betting run
        '''
        if self.curr_betting_run is None:
            self.curr_betting_run = pd.DataFrame(columns = ['Sea','LID', 'HID','AID','OddsH','OddsD','OddsA','P(H)', 'P(D)', 'P(A)','BetH','BetD','BetA'])

        existing_indexes = [match for match in data_frame.index if match in self.curr_betting_run.index] # matches that are already indexed
        existing_matches = data_frame.loc[existing_indexes]

        for i in existing_matches.index:
            for key in existing_matches.columns:
                self.matches.at[i,key] = data_frame.at[i,key] # we assume that the newer dataframe is right

        new_indexes = [match for match in data_frame.index if match not in existing_indexes] # matches to append to the self.matches as a whole
        new_matches = data_frame.loc[new_indexes]
        for i in new_matches.index:
            self.matches.loc[i] = new_matches.loc[i] # copy all of the previously  unknown matches to matches # }}}

    def __eval_matches(self, data_frame, check_for_new=True):
        ''' # {{{
        Evalutate the matches and all the values for match in the dataframe.

        Parameters:
        data_frame(pd.DataFrame): must be indexed by the 'MatchID'
        check_for_new=True(bool): for efectivity purposes... If we are adding the 'P_dis' or the 'Bets', we do not need to check for new teams.
        '''

        existing_indexes = [match for match in data_frame.index if match in self.matches.index] # matches that are already indexed
        existing_matches = data_frame.loc[existing_indexes]
        for i in existing_matches.index:
            for key in existing_matches.columns:
                self.matches.at[i,key] = data_frame.at[i,key] # we assume that the newer dataframe is right

        if check_for_new:
            new_indexes = [match for match in data_frame.index if match not in existing_indexes] # matches to append to the self.matches as a whole
            new_matches = data_frame.loc[new_indexes]
            for i in new_matches.index:
                self.matches.loc[i] = new_matches.loc[i] # copy all of the previously  unknown matches to matches}}}

    def end_data_agregation_iter(self):
        ''' # {{{
        End the current data agregation iteration. It has to be run after all the available data has been passed.
        '''
        self.curr_P_dis = self.curr_summary = self.curr_bets = self.curr_inc = self.curr_opps = self.curr_betting_run = self.curr_teams = None# }}}

        #####################################################################
        #  UPDATE THE FEATURES THAT CAN BE EXTRACTED FROM THE DATA IN SELF  #
        #####################################################################

    def update_features(self):
        '''
        Update the features for the data stored in `self`.
        '''
        self.__update_time_features()
        self.__update_season_time_features()
        self.__update_last_match_features()

    def __update_time_features(self):
        pass

    def __update_last_match_features(self):
        pass

    def __update_season_time_features(self):
        pass


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

        Returns:
        (pd.DataFrame(columns=['TEAM', 'OPPO'])) = SC == score
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

        Returns:
        (pd.DataFrame(columns=['TEAM', 'OPPO'])) = SC == score
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

