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
    # TODO: Add dtypes to the self.attributes that are dataframes for faster operations [TLE] <16-11-20, kunzaatko> #
    def __init__(self, no_copy=True, sort_columns=True):
        '''

        Parameters:
            no_copy(bool): Whether to store the `curr_opps`, `curr_inc`, `curr_P_dis`, and `curr_bets` to `self`. This will make the it run faster.
        '''

        ########################
        #  private attributes  #
        ########################
        self._no_copy = no_copy
        self._sort_columns = sort_columns
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
        # self.betting_runs = pd.DataFrame(columns = ['Sea','LID', 'HID','AID','OddsH','OddsD','OddsA','P(H)', 'P(D)', 'P(A)','BetH','BetD','BetA']) # `opps` that was passed with the associated `P_dis`, and the associated `bets` (series). Indexed by the date that it occured in opps.
        self.matches = pd.DataFrame(columns=['opps_Date','Sea','Date','Open','LID','HID','AID','HSC','ASC','H','D','A','OddsH','OddsD','OddsA','BetH','BetD','BetA']) # All matches played by IDs ﭾ


        #########################
        #  Features attributes  #
        #########################
        # 'LID = Leagues' 'SC = score (pd.DataFrame(columns=['TEAM', 'OPPO']))', 'RES = result (pd.DataFrame(columns=['TEAM', 'DRAW', 'OPPO']))', 'PLAYED = #matches_played (int)', 'NEW = new (bool)', 'ACU = accuracy (float)'
        self.team_index = pd.DataFrame(columns=['LID','LL_SC', 'LL_RES', 'LL_PLAYED', 'LL_ACCU']) # recorded teams
        self.time_data = pd.DataFrame(columns=['SL_SC', 'SL_RES', 'SL_PLAYED', 'SL_ACCU']) # data frame for storing all the time characteristics for seasons

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
            inc = inc.loc[:,~inc.columns.str.match('Unnamed')] # removing the 'Unnamed: 0' column (memory saning) See: https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe
            self._curr_inc_teams = np.unique(np.concatenate((inc['HID'].to_numpy(dtype='int64'),inc['AID'].to_numpy(dtype='int64'))))
            self._eval_inc(inc)

        if opps is not None:
            opps = opps.loc[:,~opps.columns.str.match('Unnamed')] # removing the 'Unnamed: 0' column (memory saning) See: https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe
            self._curr_opps_teams = np.unique(np.concatenate((opps['HID'].to_numpy(dtype='int64'),opps['AID'].to_numpy(dtype='int64'))))
            opps['opps_Date'] = self.today
            self._eval_opps(opps)

        if P_dis is not None:
            self._eval_P_dis(P_dis)

        if bets is not None:
            self._eval_bets(bets)

        if self._sort_columns:
            self.matches = self.matches[['opps_Date','Sea','Date','Open','LID','HID','AID','HSC','ASC','H','D','A','OddsH','OddsD','OddsA','BetH','BetD','BetA']]
            self.team_index

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
        self._eval_matches(opps)
        if not self._no_copy:
            self.curr_opps = opps
        # }}}

    def _eval_P_dis(self, P_dis):
        # {{{
        if not self._no_copy:
            self.curr_P_dis = P_dis
        # }}}

    def _eval_bets(self, bets):
        # {{{
        self._eval_matches(bets, check_for_new=False)
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
            index_self_teams = self.team_index.index.to_numpy(dtype='int64')
            # unique teams that are stored in the data frame
            index_data_frame = data_frame_teams
            # teams in the data_frame that are not stored in the self.team_index
            index_new_teams = np.setdiff1d(index_data_frame, index_self_teams)

            if not len(index_new_teams) == 0: # if there are any new teams (otherwise invalid indexing)
                # DataFrame of new teams
                new_teams = pd.DataFrame(index=index_new_teams)
                lids_frame = pd.concat((data_frame[['HID','LID']].set_index('HID'),data_frame[['AID','LID']].set_index('AID'))) # TODO: This will not work if there are multiple LIDs for one team in one inc <15-11-20, kunzaatko> # NOTE: This is probably working only because the inc already added some teams.
                lids = lids_frame[~lids_frame.index.duplicated(keep='first')].loc[index_new_teams]
                # Making a list from the 'LID's
                new_teams['LID'] = lids.apply(lambda row: np.array([row.LID]), axis=1) # this is costly but is only run once for each match %timeit dataset['LID'] = dataset.apply(lambda row: [row.LID], axis=1) -> 463 ms ± 13.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                self.team_index = pd.concat((self.team_index, new_teams))

            ##############
            #  NEW LIDS  #
            ##############

            # NOTE: This could be optimised radically but it has shown to be a pain in the ass so this is it. If there will be a 'TLE' (time limit exceeded) error, this is the place to change <15-11-20, kunzaatko> #

            # teams in the data_frame that are stored in the self.team_index (teams that could have been changed)
            index_old_teams = np.intersect1d(index_self_teams,index_data_frame)
            index_old_teams_HID = np.intersect1d(index_old_teams, data_frame['HID'].to_numpy(dtype='int64'))
            index_old_teams_AID = np.intersect1d(index_old_teams, data_frame['AID'].to_numpy(dtype='int64'))

            for index in index_old_teams_HID:
                if not type(data_frame.set_index('HID').loc[index]) == pd.DataFrame:
                    if not data_frame.set_index('HID').loc[index]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])
                else:
                    if not data_frame.set_index('HID').loc[index].iloc[0]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])

            for index in index_old_teams_AID:
                if not type(data_frame.set_index('AID').loc[index]) == pd.DataFrame:
                    if not data_frame.set_index('AID').loc[index]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])
                else:
                    if not data_frame.set_index('AID').loc[index].iloc[0]['LID'] in self.team_index.at[index,'LID']:
                        self.team_index.at[index,'LID'] = np.append(self.team_index.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])

            # see also (https://stackoverflow.com/questions/45062340/check-if-single-element-is-contained-in-numpy-array)}}}

    def _eval_matches(self, data_frame, check_for_new=True):
        # {{{
        self.matches = self.matches.combine_first(data_frame)
        # }}}

        #####################################################################
        #  UPDATE THE FEATURES THAT CAN BE EXTRACTED FROM THE DATA IN SELF  #
        #####################################################################

    def update_features(self):
        '''
        Update the features for the data stored in `self`.
        '''
        self._update_time_features()
        self._update_season_time_features()
        self._update_last_match_features()

    def _update_time_features(self):
        self._update_LL_time_features()

    def _update_LL_time_features(self):
        if self.today in self.matches['Date']:
            matches_played_today = self.matches.groupby('Date').get_group(self.today)
        else:
            matches_played_today = None
        self._update_LL_PLAYED(matches_played_today)

    def _update_LL_PLAYED(self, matches_played_today):
        if matches_played_today is not None:
            teams_played = np.concatenate((matches_played_today['HID'].to_numpy(dtype='int64'),matches_played_today['AID'].to_numpy(dtype='int64')))
            self.team_index['LL_PLAYED'].loc[teams_played] = self.team_index.reindex(teams_played)['LL_PLAYED'].apply(lambda row: pd.isnan(row) and 1 or row + 1)

    def _update_last_match_features(self):
        pass

    def _update_season_time_features(self):
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

