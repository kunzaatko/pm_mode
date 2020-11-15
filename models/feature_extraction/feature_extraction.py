import numpy as np
import pandas as pd

'''
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

class Data:
    '''
    Class for manipulating the data and extracting characteristics.

    Attributes:
        today (None): current date (`pandas._libs.tslibs.timestamps.Timestamp`)

        curr_opps (None): current iteration `opps` (`pd.DataFrame` - `Index(['Sea', 'Date', 'LID', 'HID', 'AID', 'Open', 'OddsH', 'OddsD', 'OddsA', 'BetH', 'BetD', 'BetA'], dtype='object')`)

        curr_inc (None): current iteration `inc` (`pd.DataFrame` - `Index(['Sea', 'Date', 'LID', 'HID', 'AID', 'Open', 'OddsH', 'OddsD', 'OddsA', 'HSC', 'ASC', 'H', 'D', 'A', 'BetH', 'BetD', 'BetA'], dtype='object')`)

        curr_bets (None): current iteration `bets` (`pd.DataFrame` - `Index([`BetH`, 'BetD', 'BetA'], dtype='object')`)

    '''
    # TODO: Add parameters to restrict the tracking of unneeded evaluation data elements <14-11-20, kunzaatko> #
    def __init__(self, no_copy=True):
        '''

        Parameters:
            no_copy(bool): Whether to store the `curr_opps`, `curr_inc`, `curr_P_dis`, and `curr_bets` to `self`. This will make the it run faster.
        '''

        ########################
        #  private attributes  #
        ########################
        self._no_copy = no_copy
        self._curr_inc_teams= None # teams that are in inc
        self._curr_opps_teams= None # teams that are in opps


        ########################
        #  Storage attributes  #
        ########################
        self.curr_summary = None # current `summary` ﭾ
        self.curr_bets = None # current `bets` ﭾ
        self.curr_opps = None # current `opps` ﭾ
        self.curr_inc = None # current `inc` ﭾ
        self.curr_P_dis = None # current `P_dis` ﭾ


        ##########################
        #  Essential attributes  #
        ##########################
        self.today = None # current date
        self.bankroll = None # current bankroll
        self.betting_runs = pd.DataFrame(columns = ['Sea','LID', 'HID','AID','OddsH','OddsD','OddsA','P(H)', 'P(D)', 'P(A)','BetH','BetD','BetA']) # `opps` that was passed with the associated `P_dis`, and the associated `bets` (series). Indexed by the date that it occured in opps.
        self.matches = pd.DataFrame(columns=['Date', 'Sea','LID','HID','AID','OddsH','OddsD','OddsA','HSC','ASC','H','D','A','BetH','BetD','BetA']) # All matches played by IDs ﭾ

        #########################
        #  Features attributes  #
        #########################
        # 'LID = Leagues' 'SC = score (pd.DataFrame(columns=['TEAM', 'OPPO']))', 'RES = result (pd.DataFrame(columns=['TEAM', 'DRAW', 'OPPO']))', 'PLAYED = #matches_played (int)', 'NEW = new (bool)', 'ACU = accuracy (float)'
        self.team_index = pd.DataFrame(columns=['LL_SC', 'LL_RES', 'LL_PLAYED', 'LL_NEW', 'LL_ACCU']) # recorded teams
        self.time_data = pd.DataFrame(columns=['SL_SC', 'SL_RES', 'SL_PLAYED', 'SL_NEW', 'SL_ACCU']) # data frame for storing all the time characteristics for seasons

        # 'SC = score (TEAM, OPPO)', 'RES = result (pd.DataFrame(columns=['TEAM', 'DRAW', 'OPPO']))', 'DATE = date', 'LM_SIDE = home/away (str)', 'LM_P_DIS = pd.DataFrame(columns=['win_p', 'draw_p', 'lose_p'])'
        self.last_match_data = pd.DataFrame(columns=['MatchID', 'LM_SC (T,O)', 'LM_RES (T,D,O)', 'LM_DATE', 'LM_SIDE (H,A)', 'LM_P_DIS (W,D,L)']) # data frame for storing all the Last-match characteristics
        self.matches_data = pd.DataFrame(columns=['M_DATA_FRAME']) # data frame for moving the data of the last match when a new match is played


    ######################################
    #  UPDATING THE DATA STORED IN SELF  #
    ######################################

    def update_data(self, opps=None ,summary=None, inc=None, P_dis=None, bets=None):
        # {{{
        '''
        Run the iteration update of the data stored.
        ! Summary has to be updated first to get the right date!

        Parameters:
        All the parameters are supplied by the evaluation loop.
        opps(pandas.DataFrame): dataframe that includes the opportunities for betting.
        summary(pandas.DataFrame): includes the `Max_bet`, `Min_bet` and `Bankroll`.
        inc(pandas.DataFrame): includes the played matches with the scores for the model.
        '''
        if summary is not None:
            self._eval_summary(summary)

        if inc is not None:
            self._curr_inc_teams = np.unique(np.concatenate((inc['HID'].to_numpy(dtype='int64'),inc['AID'].to_numpy(dtype='int64'))))
            self._eval_inc(inc)

        if opps is not None:
            self._curr_opps_teams = np.unique(np.concatenate((opps['HID'].to_numpy(dtype='int64'),opps['AID'].to_numpy(dtype='int64'))))
            self._eval_opps(opps)

        if P_dis is not None:
            self._eval_P_dis(P_dis)

        if bets is not None:
            self._eval_bets(bets)

        # }}}

    def _eval_summary(self, summary):
        # {{{
        self.today = summary['Date'][0]
        self.bankroll = summary['Bankroll'][0]

        if not self._no_copy:
            self.curr_summary = summary
        # }}}

    def _eval_inc(self, inc):
        # {{{
        self._eval_teams(inc, self._curr_inc_teams)
        self._eval_matches(inc)

        if not self._no_copy:
            self.curr_inc = inc
        # }}}

    def _eval_opps(self, opps):
        # {{{
        self._eval_teams(opps, self._curr_inc_teams)
        # self._eval_matches(opps)
        # self._eval_betting_run(opps)

        if not self._no_copy:
            self.curr_opps = opps
        # }}}

    def _eval_P_dis(self, P_dis):
        # {{{
        self._eval_betting_run(P_dis)

        if not self._no_copy:
            self.curr_P_dis = P_dis
        # }}}

    def _eval_bets(self, bets):
        # {{{
        self._eval_matches(bets, check_for_new=False)
        self._eval_betting_run(bets)

        if not self._no_copy:
            self.curr_bets = bets
        # }}}

    def _eval_teams(self, data_frame, data_frame_teams):
        # {{{

        if not data_frame.empty:

            ###############
            #  NEW TEAMS  #
            ###############

            # teams that are already stored in the self.team_index
            index_teams = self.team_index.index.to_numpy(dtype='int64')
            # unique teams that are stored in the data frame
            # data_frame_ID_LID = np.concatenate((data_frame[['HID','LID']].to_numpy(),data_frame[['AID','LID']].to_numpy()))
            data_frame_teams_indexes = data_frame_teams
            # teams in the data_frame that are not stored in the self.team_index
            new_teams_index = np.setdiff1d(data_frame_teams_indexes, index_teams)

            if not len(new_teams_index) == 0: # if there are any new teams (otherwise invalid indexing)
                # new_teams_index = np.sort(data_frame_teams_indexes[new_teams_boolean_index])
                # DataFrame of new teams
                new_teams = pd.DataFrame(index=new_teams_index)
                lids = pd.concat((data_frame[['HID','LID']].set_index('HID'),data_frame[['AID','LID']].set_index('AID'))).loc[new_teams_index] # TODO: This will not work if there are multiple LIDs for one team in one inc <15-11-20, kunzaatko> #
                # Making a list from the 'LID's
                new_teams['LID'] = lids.apply(lambda row: np.array([row.LID]), axis=1) # this is costly but is only run once for each match %timeit dataset['LID'] = dataset.apply(lambda row: [row.LID], axis=1) -> 463 ms ± 13.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                self.team_index = pd.concat((self.team_index, new_teams))

            ##############
            #  NEW LIDS  #
            ##############

            # NOTE: This could be optimised radically but it has shown to be a pain in the ass so this is it. If there will be a 'TLE' (time limit exceeded) error, this is the place to change <15-11-20, kunzaatko> #

            # teams in the data_frame that are stored in the self.team_index (teams that could have been changed)
            old_teams_index = np.intersect1d(index_teams,data_frame_teams_indexes)
            old_teams_index_HID = np.intersect1d(old_teams_index, data_frame['HID'].to_numpy(dtype='int64'))
            old_teams_index_AID = np.intersect1d(old_teams_index, data_frame['AID'].to_numpy(dtype='int64'))

            # lids = pd.concat((data_frame[['HID','LID']].set_index('HID'),data_frame[['AID','LID']].set_index('AID')))
            # lids['ID'] = lids.index
            # lids = lids.drop_duplicates(subset=['ID'])

            for index in old_teams_index_HID:
                if not type(data_frame.set_index('HID').loc[index]) == pd.DataFrame:
                    if not data_frame.set_index('HID').loc[index]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])
                else:
                    if not data_frame.set_index('HID').loc[index].iloc[0]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])

            for index in old_teams_index_AID:
                if not type(data_frame.set_index('AID').loc[index]) == pd.DataFrame:
                    if not data_frame.set_index('AID').loc[index]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])
                else:
                    if not data_frame.set_index('AID').loc[index].iloc[0]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])

            # changed_old_teams = [ID for (ID,LID,LIDs) in zip(old_teams_index, data_frame.reindex(old_teams_index)['LID'].to_numpy(), self.team_index.reindex(old_teams_index)['LID'].to_numpy()) if LID in LIDs]
            # self.team_index.reindex(changed_old_teams)['LID'] = self.team_index.reindex(changed_old_teams).apply(lambda row: np.append(row.LID, data_frame.at[row.name,'LID']))
            # print(self.team_index.reindex(changed_old_teams).apply(lambda row: np.append(row.LID, data_frame.at[row.name,'LID'])))

            # see also (https://stackoverflow.com/questions/45062340/check-if-single-element-is-contained-in-numpy-array)}}}

    def _eval_matches(self, data_frame, check_for_new=True):
        # {{{
        # TODO: change this to concatenation <15-11-20, kunzaatko> #
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

    def _eval_betting_run(self, data_frame):
        # {{{
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
            self.matches.loc[i] = new_matches.loc[i] # copy all of the previously  unknown matches to matches
        # }}}

    # DEPRECATED
    def end_data_agregation_iter(self):
        ''' # {{{
        End the current data agregation iteration. It has to be run after all the available data has been passed.
        '''
        self.today = self.curr_P_dis = self.curr_summary = self.curr_bets = self.curr_inc = self.curr_opps = self.curr_betting_run  = None# }}}

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

