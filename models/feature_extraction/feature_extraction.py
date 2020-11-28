import numpy as np
import pandas as pd
from numba import njit

class Data:
    '''
    Class for manipulating the data and extracting characteristics.

    Attributes:
        today (pd.datetime64): current date (`pandas._libs.tslibs.timestamps.Timestamp`)
        self.bankroll (int): bankroll from the summary
    '''

    # TODO: Make everything that is possible inplace and copy=False to increase performance
    # TODO: Add dtypes to the self.attributes that are dataframes for faster operations [TLE] <16-11-20, kunzaatko> #
    def __init__(self, sort_columns=True, optional_data_cols=[], ELO_mean_ELO=1500, ELO_k_factor=20, LL_data = True, dummy=False):
    # {{{
        '''
        Parameters:
            sort_columns(True): Sort the columns of the dataframes
            optional_data_cols(list(str)): possible values:
                'ELO_rating' - calculate the ELO rating as a feature in the LL_data DataFrame
            ELO_mean_ELO(int): ELO, that teams start with
            ELO_k_factor(int): maximum ELO points exchanged in one match
        '''

        ########################
        #  private attributes  #
        ########################
        self._sort_columns = sort_columns
        self._curr_inc_teams = None # teams that are in inc
        self._curr_opps_teams = None # teams that are in opps
        self._matches_not_registered_to_features = None # matches, that were not yet counted into team features
        self._dummy = dummy

        if 'ELO_rating' in optional_data_cols:
            self.ELO_rating = True
            self.ELO_mean_ELO = ELO_mean_ELO
            self.ELO_k_factor = ELO_k_factor
        else:
            self.ELO_rating = False

        ########################
        #  Storage attributes  #
        ########################
        self.yesterday = None # this is used for initialization in very first inc of data and then as reference to yesterday
        self.today = None # current date
        self.bankroll = None # current bankroll
        self.opps_matches = None

        ##########################
        #  Essential attributes  #
        ##########################
        # FIXME: The 'opps_Date' column is does not work, since we get multiple the matches with the same ID for several consecutive days <16-11-20, kunzaatko> #
        # FIXME: Also the P_dis that is evaluated by our model can change from day to day so the P_dis, that we have stored is only the last one <16-11-20, kunzaatko> #

        # `self.matches`
        # index    || 'opps_Date'            | 'Sea'  | 'Date'       | 'Open'                      | 'LID'           | 'HID'        | 'AID'
        # match ID || date of opps occurence | season | date of play | date of betting possibility | league ID (str) | home team ID | away team ID
        #           | 'HSC'             | 'ASC'             | 'H'      | 'D'  | 'A'      | 'OddsH'          | 'OddsD'      | 'OddsA'
        #           | home goals scored | away goals scored | home win | draw | away win | odds of home win | odds of draw | odds of away win
        #           | 'P(H)'               | 'P(D)'           | 'P(A)'               | 'BetH'       | 'BetD'   | 'BetA'
        #           | model prob. home win | model prob. draw | model prob. away win | bet home win | bet draw | bet away win

        types = {'Date':'datetime64[ns]', 'Sea':'int16','HID':'int16','AID':'int16','OddsH':'float64','OddsD':'float64','OddsA':'float64','HSC':'int16','ASC':'int16','H':'int64','D':'int64','A':'int64','P(H)':'float64','P(D)':'float64','P(A)':'float64','BetH':'float64','BetD':'float64','BetA':'float64'}
        self.matches = pd.DataFrame(columns=['opps_Date','Sea','Date','LID','HID','AID','HSC','ASC','H','D','A','OddsH','OddsD','OddsA','P(H)','P(D)', 'P(A)','BetH','BetD','BetA']).astype(types, copy=False) # All matches played by IDs ﭾ


        #########################
        #  Features attributes  #
        #########################

        # `self.LL_data`
        # LL: life-long
        # index   || 'LID'            | 'LL_Goals_Scored' | 'LL_Goals_Conceded' | 'LL_Wins' | 'LL_Draws' | 'LL_Loses'
        # team ID || league ID (list) | goals scored      | goals conceded      | wins      | draws      | loses
        #          | 'LL_Played'    | 'LL_Accu'
        #          | played matches | model accuracy
        self.LL_data = pd.DataFrame(columns=['LID','LL_Goals_Scored','LL_Goals_Conceded','LL_Wins', 'LL_Draws', 'LL_Loses', 'LL_Played', 'LL_Accu']) # recorded teams

        if self.ELO_rating:
            self.LL_data['ELO_rating'] = np.nan

        # `self.SL_data`
        # SL: season-long
        # index (multiindex)|| 'LID'            | 'SL_Goals_Scored' | 'SL_Goals_Conceded' | 'SL_Wins' | 'SL_Draws' | 'SL_Loses'
        # season,team ID    || league ID (list) | goals scored      | goals conceded      | wins      | draws      | loses
        #                    | 'SL_Played'    | 'SL_Accu'
        #                    | played matches | model accuracy
        self.SL_data = pd.DataFrame(columns=['LID','SL_Goals_Scored', 'SL_Goals_Conceded', 'SL_Wins', 'SL_Draws', 'SL_Loses', 'SL_Played', 'SL_Accu']) # data frame for storing all the time characteristics for seasons


        # `self.match_data`
        # index   || 'MatchID' | 'Sea'  | 'Date'       | 'Oppo'      | 'Home'       | 'Away'       | 'M_Goals_Scored' | 'M_Goals_Conceded'
        # team ID || match ID  | season | date of play | opponent id | team is home | team is away | goals scored     | goals conceded
        #          | 'M_Win'   | 'M_Draw'  | 'M_Lose'   | 'M_P(Win)'      | 'M_P(Draw)'      | 'M_P(Lose)'      | 'M_Accu'
        #          | match win | match draw| match lose | model prob. win | model prob. draw | model prob. lose | model accuracy
        self.match_data = pd.DataFrame(columns=['MatchID', 'Date' , 'Oppo', 'Home', 'Away',  'M_Goals_Scored', 'M_Goals_Conceded', 'M_Win','M_Draw', 'M_Lose','M_P(Win)','M_P(Draw)', 'M_P(Lose)','M_Accu'])

        # `self.features`
        # index || 'H_GS_GC_diff_#5'|'A_GS_GC_diff_#5' | 'H_GS_#' | 'A_GS_#' | 'H_GC_#' | 'A_GC_#' | 'H_WR_#' | 'A_WR_#' | 'H_DR_#' | 'A_DR_#' | 'H_LR_#' | 'A_Lr_#' |
        # MatchID || goals scored - goals conceded in last 15 matches for home team | goals scored - goals conceded difference in last 15 matches for away team | home goals scored in last # matches | away goals scored in last # matches | home win rate in last # matches | away lose rate in last # matches | home draw rate in last # matches | away draw rate in last # matches | home lose rate in last # matches | away lose rate in last # matches |
        self.features = pd.DataFrame(columns=['H_GS_GC_diff_#15','A_GS_GC_diff_#15','H_GS_#','A_GS_#','H_GC_#','A_GC_#','H_WR_#','A_WR_#','H_DR_#','A_DR_#','H_LR_#','A_LR_#'])
    # }}}

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
        if not self._dummy:
            if summary is not None:
                self._EVAL_summary(summary)

            if inc is not None:
                if self.today in inc['Date'].values:
                    print(all(pd.isna(inc.groupby('Date').get_group(self.today))))
                inc = inc.loc[:,~inc.columns.str.match('Unnamed')] # removing the 'Unnamed: 0' column (memory saning) See: https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe
                self._curr_inc_teams = np.unique(np.concatenate((inc['HID'].to_numpy(dtype='int64'),inc['AID'].to_numpy(dtype='int64'))))
                self._EVAL_inc(inc)

            if opps is not None:
                opps = opps.loc[:,~opps.columns.str.match('Unnamed')] # removing the 'Unnamed: 0' column (memory saning) See: https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe
                self._curr_opps_teams = np.unique(np.concatenate((opps['HID'].to_numpy(dtype='int64'),opps['AID'].to_numpy(dtype='int64'))))
                opps['opps_Date'] = self.today
                self._EVAL_opps(opps)

            if P_dis is not None:
                self._EVAL_P_dis(P_dis)

            if bets is not None:
                self._EVAL_bets(bets)

            if self._sort_columns:
                self.matches = self.matches[['opps_Date','Sea','Date','LID','HID','AID','HSC','ASC','H','D','A','OddsH','OddsD','OddsA','P(H)','P(D)', 'P(A)','BetH','BetD','BetA']]
                if self.ELO_rating:
                    self.LL_data = self.LL_data[['LID', 'LL_Goals_Scored','LL_Goals_Conceded','LL_Wins', 'LL_Draws', 'LL_Loses', 'LL_Played', 'LL_Accu','ELO_rating']]
                else:
                    self.LL_data = self.LL_data[['LID', 'LL_Goals_Scored','LL_Goals_Conceded','LL_Wins', 'LL_Draws', 'LL_Loses', 'LL_Played', 'LL_Accu']]

        # }}}

    def _EVAL_summary(self, summary):
        # {{{
        self.today = summary['Date'][0]
        self.yesterday = self.today - pd.DateOffset(1) # -> We do not have to worry about self.yesterday being None anymore
        self.bankroll = summary['Bankroll'][0]
        # }}}

    def _EVAL_inc(self, inc):
        # {{{
        self._eval_teams(inc, self._curr_inc_teams)
        self._eval_matches(inc,update_columns=['HSC','ASC','H','D','A'])

        if self.ELO_rating:
            self._eval_inc_update_ELO(inc)
        # }}}

    def _eval_inc_update_ELO(self, inc):
    # {{{
        '''
        Update the ELO ratings for the new incremented data.
        '''
        def elo_for_one_team(row):
            Home_ID,Away_ID,Home_win,_,Away_win = row.HID,row.AID,row.H,row.D,row.A
            [Home_elo, Away_elo] = [self.LL_data.at[ID,'ELO_rating'] for ID in [Home_ID,Away_ID]]
            [Home_expected, Away_expected] = [1/(1+10**((elo_1 - elo_2) / 400)) for (elo_1, elo_2) in [(Away_elo, Home_elo), (Home_elo, Away_elo)]]
            if any([Home_win,Away_win]):
                self.LL_data.at[Home_ID, 'ELO_rating'] += self.ELO_k_factor * (Home_win - Home_expected)
                self.LL_data.at[Away_ID, 'ELO_rating'] += self.ELO_k_factor * (Away_win - Away_expected)

        inc.apply(elo_for_one_team,axis=1)
    # }}}

    def _EVAL_opps(self, opps):
        # {{{
        self.opps_matches = opps.index.to_numpy()
        self._eval_teams(opps, self._curr_inc_teams)
        self._eval_matches(opps, update_columns=['Sea','Date','LID','HID','AID','OddsH','OddsA','OddsD'])
        # }}}

    def _EVAL_P_dis(self, P_dis):
        # {{{
        self._eval_matches(P_dis,update_columns=['P(H)', 'P(D)', 'P(A)'])
        # }}}

    def _EVAL_bets(self, bets):
        # {{{
        self._eval_matches(bets,update_columns=['BetH','BetD','BetA'])
        # }}}

    def _eval_teams(self, data_frame, data_frame_teams):
        # {{{

        if not data_frame.empty:

            ###############
            #  NEW TEAMS  #
            ###############

            # teams that are already stored in the self.LL_data
            index_self_teams = self.LL_data.index.to_numpy(dtype='int64')
            # unique teams that are stored in the data frame
            index_data_frame = data_frame_teams
            # teams in the data_frame that are not stored in the self.LL_data
            index_new_teams = np.setdiff1d(index_data_frame, index_self_teams)

            if not len(index_new_teams) == 0: # if there are any new teams (otherwise invalid indexing)
                # DataFrame of new teams
                new_teams = pd.DataFrame(index=index_new_teams)
                # lids_frame = pd.concat((data_frame[['HID','LID']].set_index('HID'),data_frame[['AID','LID']].set_index('AID'))) # TODO: This will not work if there are multiple LIDs for one team in one inc <15-11-20, kunzaatko> # NOTE: This is probably working only because the inc already added some teams.
                # lids = lids_frame[~lids_frame.index.duplicated(keep='first')].loc[index_new_teams]
                # Making a list from the 'LID's
                # new_teams['LID'] = lids.apply(lambda row: np.array([row.LID]), axis=1) # this is costly but is only run once for each match %timeit dataset['LID'] = dataset.apply(lambda row: [row.LID], axis=1) -> 463 ms ± 13.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                if self.ELO_rating:
                    new_teams['ELO_rating'] = self.ELO_mean_ELO
                self.LL_data = pd.concat((self.LL_data, new_teams))
                self.LL_data.fillna(0., inplace=True)

            ##############
            #  NEW LIDS  #
            ##############

            # NOTE: This could be optimised radically but it has shown to be a pain in the ass so this is it. If there will be a 'TLE' (time limit exceeded) error, this is the place to change <15-11-20, kunzaatko> #

            # teams in the data_frame that are stored in the self.LL_data (teams that could have been changed)
            # index_old_teams = np.intersect1d(index_self_teams,index_data_frame)
            # index_old_teams_HID = np.intersect1d(index_old_teams, data_frame['HID'].to_numpy(dtype='int64'))
            # index_old_teams_AID = np.intersect1d(index_old_teams, data_frame['AID'].to_numpy(dtype='int64'))

            # for index in index_old_teams_HID:
            #     if not type(data_frame.set_index('HID').loc[index]) == pd.DataFrame:
            #         if not data_frame.set_index('HID').loc[index]['LID'] in self.LL_data.at[index,'LID']:
            #             self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])
            #     else:
            #         if not data_frame.set_index('HID').loc[index].iloc[0]['LID'] in self.LL_data.at[index,'LID']:
            #             self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])

            # for index in index_old_teams_AID:
            #     if not type(data_frame.set_index('AID').loc[index]) == pd.DataFrame:
            #         if not data_frame.set_index('AID').loc[index]['LID'] in self.LL_data.at[index,'LID']:
            #             self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])
            #     else:
            #         if not data_frame.set_index('AID').loc[index].iloc[0]['LID'] in self.LL_data.at[index,'LID']:
            #             self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])

            # see also (https://stackoverflow.com/questions/45062340/check-if-single-element-is-contained-in-numpy-array)}}}

    # TODO: Probably does not work correctly for the bets. The bets should not be combined for the `opps` and the `inc` but only for the `bets` dataframe. <17-11-20, kunzaatko> #
    # TODO: the 'opps_Date' is not working. The indexes should not be concatenated but appended for new matches if they do not have the same 'opps_Date'... (When they are not added on the same day) <17-11-20, kunzaatko> # -> the problem with this is though that we would have to groupby matchid to to access a match, and multiple MatchIDs would be the same in the dataframe -> We should consider adding a new frame with this data (or maybe the bets should be recorded as an associated series of the match... What is your oppinion/solution?
    def _eval_matches(self, data_frame,update_columns=[]):
        # {{{
        # !!! this changes the dtypes and therefore runs slowly (as per https://github.com/pandas-dev/pandas/issues/28613)
        # self.matches = self.matches.combine_first(data_frame)

        old_matches = np.intersect1d(data_frame.index.to_numpy(dtype='int32'),self.matches.index.to_numpy(dtype='int32'))
        self.matches.update(data_frame[update_columns].loc[old_matches])
        new_matches = np.setdiff1d(data_frame.index.to_numpy(dtype='int32'),old_matches)
        # if there are no such indices, then append whole frame
        self.matches = self.matches.append(data_frame.loc[new_matches]).sort_index()
        # }}}

        #####################################################################
        #  UPDATE THE FEATURES THAT CAN BE EXTRACTED FROM THE DATA IN SELF  #
        #####################################################################

    def update_features(self):
    # {{{
        '''
        Update the features for the data stored in `self`.
        '''
        if not self._dummy:
            self._UPDATE_LL_data_features()
            #self._UPDATE_SL_data_features()
            self._UPDATE_match_data_features()
            self._UPDATE_features()
    # }}}

    def _UPDATE_LL_data_features(self):
    # {{{
        '''
        Populate all the features from the frame `self.LL_data`
        '''
        matches_played_before = self.matches[self.matches['Date'] < self.today] if self.yesterday is None else \
            self.matches.groupby('Date').get_group(self.yesterday) if self.yesterday in self.matches['Date'].to_numpy() \
            else None
        matches_played_today = self.matches.groupby('Date').get_group(self.today) if self.today in self.matches['Date'].to_numpy() \
                else None

        self._update_LL_Played(matches_played_before)
        self._update_LL_Goals(matches_played_before)
        self._update_LL_Res(matches_played_before)
        self._update_LL_Accu(matches_played_before)
    # }}}

    def _update_LL_Played(self, matches_played):
    # {{{
        '''
        Update 'LL_Played' (games) of the fram self.LL_data
        :param matches_played: pd.Dataframe:
            Contains matches played at self.yesterday
        '''
        if matches_played is not None:
            teams_played = np.unique(np.concatenate((matches_played['HID'].to_numpy(dtype='int64'),
                                                     matches_played['AID'].to_numpy(dtype='int64'))), return_counts=True)
            self.LL_data.loc[teams_played[0], 'LL_Played'] = self.LL_data.loc[teams_played[0], 'LL_Played'] + \
                                                             teams_played[1]
    # }}}

    def _update_LL_Goals(self, matches_played):
    # {{{
        '''
        Update 'LL_Goals_Scored' and 'LL_Goals_Conceded' of the frame `self.LL_data`
        '''
        if matches_played is not None:
            teams_goals_scored = np.concatenate([matches_played[['HID', 'HSC']].to_numpy(dtype='int64'),
                                                 matches_played[['AID', 'ASC']].to_numpy(dtype='int64')])
            teams_goals_conceded = np.concatenate([matches_played[['HID', 'ASC']].to_numpy(dtype='int64'),
                                                   matches_played[['AID', 'HSC']].to_numpy(dtype='int64')])

            scored = fast(teams_goals_scored)
            conceded = fast(teams_goals_conceded)
            self.LL_data.loc[scored[:, 0], 'LL_Goals_Scored'] = \
                self.LL_data.loc[scored[:, 0], 'LL_Goals_Scored'] + scored[:, 1]
            self.LL_data.loc[conceded[:, 0], 'LL_Goals_Conceded'] = \
                self.LL_data.loc[conceded[:, 0], 'LL_Goals_Conceded'] + conceded[:, 1]
    # }}}

    def _update_LL_Res(self, matches_played):
    # {{{
        '''
        Update 'LL_Wins', 'LL_Draws' and 'LL_Loses' of the frame `self.LL_data`
        '''
        if matches_played is not None:
            teams_wins = np.concatenate([matches_played[['HID', 'H']].to_numpy(dtype='int64'),
                                                 matches_played[['AID', 'A']].to_numpy(dtype='int64')])
            teams_loses = np.concatenate([matches_played[['HID', 'A']].to_numpy(dtype='int64'),
                                                   matches_played[['AID', 'H']].to_numpy(dtype='int64')])
            teams_draws = np.concatenate([matches_played[['HID', 'D']].to_numpy(dtype='int64'),
                                                   matches_played[['AID', 'D']].to_numpy(dtype='int64')])

            wins = fast(teams_wins)
            loses = fast(teams_loses)
            draws = fast(teams_draws)

            self.LL_data.loc[wins[:, 0], 'LL_Wins'] = self.LL_data.loc[wins[:, 0], 'LL_Wins'] + wins[:, 1]
            self.LL_data.loc[loses[:, 0], 'LL_Loses'] = self.LL_data.loc[loses[:, 0], 'LL_Loses'] + loses[:, 1]
            self.LL_data.loc[draws[:, 0], 'LL_Draws'] = self.LL_data.loc[draws[:, 0], 'LL_Draws'] + draws[:, 1]
    # }}}

    def _update_LL_Accu(self, matches_played):
    # {{{
        '''
        Update 'LL_Accu' of the frame `self.LL_data`
        '''
        if matches_played is not None:
            pass
    # }}}

    def _UPDATE_SL_data_features(self):
    # {{{
        '''
        Populate all the features of `self.SL_data`
        '''
        # TODO: should be done incrementaly <17-11-20, kunzaatko> #
        # TODO I assume that 'self.SL_data' are updated when new team will be present in 'inc' (Many98)
        matches_played_before = self.matches[self.matches['Date'] < self.today] if self.yesterday is None else \
            self.matches.groupby('Date').get_group(self.yesterday) if self.yesterday in self.matches['Date'].to_numpy() \
            else None

        self._update_SL_Goals(matches_played_before)
        self._update_SL_Res(matches_played_before)
        self._update_SL_Played(matches_played_before)
        self._update_SL_Accu(matches_played_before)
    # }}}

    # TODO: Could be unified with `_update_LL_Goals` as `_update_Goals` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Goals(self, matches_played):
    # {{{
        '''
        Update 'SL_Goals_Scored' and 'SL_Goals_Conceded' of the frame `self.SL_data`
        '''
        if matches_played is not None:
            seasons = [season for season in matches_played.groupby('Sea')]
            for sea, season in seasons:
                teams_goals_scored = np.concatenate([season[['HID', 'HSC']].to_numpy(dtype='int64'),
                                                     season[['AID', 'ASC']].to_numpy(dtype='int64')])
                teams_goals_conceded = np.concatenate([season[['HID', 'ASC']].to_numpy(dtype='int64'),
                                                       season[['AID', 'HSC']].to_numpy(dtype='int64')])

                scored = fast(teams_goals_scored)
                conceded = fast(teams_goals_conceded)

                ind_gs = [(sea, team_id) for team_id in scored[:, 0]]
                ind_gc = [(sea, team_id) for team_id in conceded[:, 0]]

                self.SL_data.loc[ind_gs, 'SL_Goals_Scored'] = \
                    self.SL_data.loc[ind_gs, 'SL_Goals_Scored'] + scored[:, 1]
                self.SL_data.loc[ind_gc, 'SL_Goals_Conceded'] = \
                    self.SL_data.loc[ind_gc, 'SL_Goals_Conceded'] + conceded[:, 1]
    # }}}

    # TODO: Could be unified with `_update_LL_Res` as `_update_Res` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Res(self, matches_played):
    # {{{
        if matches_played is not None:
            seasons = [season for season in matches_played.groupby('Sea')]
            for sea, season in seasons:
                teams_wins = np.concatenate([season[['HID', 'H']].to_numpy(dtype='int64'),
                                             season[['AID', 'A']].to_numpy(dtype='int64')])
                teams_loses = np.concatenate([season[['HID', 'A']].to_numpy(dtype='int64'),
                                              season[['AID', 'H']].to_numpy(dtype='int64')])
                teams_draws = np.concatenate([season[['HID', 'D']].to_numpy(dtype='int64'),
                                              season[['AID', 'D']].to_numpy(dtype='int64')])

                wins = fast(teams_wins)
                loses = fast(teams_loses)
                draws = fast(teams_draws)

                ind_wins = [(sea, team_id) for team_id in wins[:, 0]]
                ind_loses = [(sea, team_id) for team_id in loses[:, 0]]
                ind_draws = [(sea, team_id) for team_id in draws[:, 0]]

                self.SL_data.loc[ind_wins, 'SL_Wins'] = \
                    self.SL_data.loc[ind_wins, 'SL_Wins'] + wins[:, 1]
                self.SL_data.loc[ind_loses, 'SL_Loses'] = \
                    self.SL_data.loc[ind_loses, 'SL_Loses'] + loses[:, 1]
                self.SL_data.loc[ind_draws, 'SL_Draws'] = \
                    self.SL_data.loc[ind_draws, 'SL_Draws'] + draws[:, 1]
    # }}}

    # TODO: Could be unified with `_update_LL_Played` as `_update_Played` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Played(self, matches_played):
    # {{{
        if matches_played is not None:
            seasons = [season for season in matches_played.groupby('Sea')]
            for sea, season in seasons:
                teams_played = np.unique(np.concatenate((season['HID'].to_numpy(dtype='int64'),
                                                         season['AID'].to_numpy(dtype='int64'))), return_counts=True)
                ind_teams = [(sea, team_id) for team_id in teams_played[0]]

                self.SL_data.loc[ind_teams, 'SL_Played'] = self.SL_data.loc[ind_teams, 'SL_Played'] + \
                                                                 teams_played[1]
    # }}}

    # TODO: Could be unified with `_update_LL_Accu` as `_update_Accu` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Accu(self, matches_played):
    # {{{
        '''
        Update 'SL_Accu' of the frame `self.LL_data`
        '''
        pass
    # }}}

    def _UPDATE_match_data_features(self):
    # {{{
        '''
        Populate all the features of `self.match_data`
        '''
        # if we are on the first inc or we skiped some dates...
        if not np.setdiff1d(self.matches.Date.to_numpy()[self.matches.Date.to_numpy() < self.yesterday],self.match_data.Date.to_numpy()).size == 0:
            self._update_add_matches(self.matches[self.matches.Date <= self.yesterday])
        elif self.yesterday in self.matches['Date'].to_numpy():
            # a dataframe of all the todays matches (matches that where played on `self.today`)
            matches_played_yesterday = self.matches.groupby('Date').get_group(self.yesterday)
            self._update_add_matches(matches_played_yesterday)

        # TODO: should be done incrementaly <17-11-20, kunzaatko> #
    # }}}

    # FIXME: does not update the matches that are not gone through at today... The matches in the first inc. <18-11-20, kunzaatko> #
    def _update_add_matches(self, matches_played_yesterday):
    # {{{
        '''
        Add the matches that were played yesterday. The fields 'MatchID', 'Date' == self.yesterday, 'Oppo' == HID/AID, 'Home' & 'Away' (int 1/0), 'M_Goals_Scored' & 'M_Goals_Conceded' (int), 'M_Win' & 'M_Draw' & 'M_Lose' (int 1/0), 'M_P(Win)' & 'M_P(Draw)' & 'M_P(Lose)' (float), 'M_Accu' should be filled.
        '''
        # the matches that played as home
        matches_home = matches_played_yesterday.set_index('HID').drop(labels=['opps_Date'],axis=1)
        renames = {'AID':'Oppo', 'HSC':'M_Goals_Scored', 'ASC':'M_Goals_Conceded', 'H':'M_Win', 'D':'M_Draw', 'A':'M_Lose', 'P(H)':'M_P(Win)', 'P(D)':'M_P(Draw)', 'P(A)':'M_P(Lose)'}
        matches_home.rename(renames, axis=1, inplace=True)
        matches_home['Home'] = 1
        matches_home['Away'] = 0
        matches_home['MatchID'] = matches_played_yesterday.index
        # TODO: Model accuracy <17-11-20, kunzaatko> #

        # the matches that played as away
        matches_away = matches_played_yesterday.set_index('AID').drop(labels=['opps_Date'],axis=1)
        renames = {'HID':'Oppo', 'ASC':'M_Goals_Scored', 'HSC':'M_Goals_Conceded', 'A':'M_Win', 'D':'M_Draw', 'H':'M_Lose', 'P(A)':'M_P(Win)', 'P(D)':'M_P(Draw)', 'P(H)':'M_P(Lose)'}
        matches_away.rename(renames, axis=1, inplace=True)
        matches_away['Home'] = 0
        matches_away['Away'] = 1
        matches_away['MatchID'] = matches_played_yesterday.index
        # TODO: Model accuracy <17-11-20, kunzaatko> #

        # TODO: Do not create a new object but only concat. <17-11-20, kunzaatko> #
        self.match_data = self.match_data.append([matches_away, matches_home])
    # }}}

    ##############
    #  FEATURES  #
    ##############

    def total_goals_to_match(self, ID, number_of_matches, MatchID=None, goal_type='scored'):
    # {{{
        '''
        Total life-long goal to match ratio.

        Parameters:
            ID(int): team id
            number_of_matches(int): num
            goal_type(str): 'scored'/'conceded'

        Returns:
            float: scored goals / # matches
        '''
        pass
    # }}}

    # TODO features working with goals_scored/conceded for particluar team should be wrapped to one method
    def goals_difference_to_num_matches(self, team_id, num_matches=1):
    # {{{
        """
        Calculates (GS-GC) of specific team from goals scored and conceded in particular number of matches played before.
        This feature should somehow aggregate information about team attack and defensive strength.
        :param team_id: int:
            Specifies particular team
        :param num_matches: int:
            Specifies particular number of matches from which the goals characteristics should be included.
            Default is set to 1.
        :return: int:

        """
        if type(num_matches) is not int or num_matches == 0:
            num_matches = 1

        # this is fastest selecting in compared with concat and append
        # %timeit matches[(matches["HID"] == team_id) | (matches["AID"] == team_id)].sort_index()
        # 1.21 ms ± 14.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        # %timeit pd.concat([matches[matches["HID"] == team_id], matches[matches["AID"] == team_id]]).sort_index()
        # 3.26 ms ± 62.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        # %timeit matches[matches["HID"]==team_id].append(matches[matches["AID"]==team_id]).sort_index()
        # 3.31 ms ± 75.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        matches_containing_team = self.matches[(self.matches["HID"] == team_id) |
                                               (self.matches["AID"] == team_id)].sort_index()[-num_matches-1:-1]
        goals_conceded, goals_scored = np.nan, np.nan
        if not matches_containing_team.empty:
            goals_conceded = matches_containing_team[matches_containing_team["HID"] == team_id]['ASC'].sum() + \
                             matches_containing_team[matches_containing_team["AID"] == team_id]['HSC'].sum()
            goals_scored = matches_containing_team[matches_containing_team["HID"] == team_id]['HSC'].sum() + \
                           matches_containing_team[matches_containing_team["AID"] == team_id]['ASC'].sum()

        return goals_scored - goals_conceded
    # }}}

    def goals_difference_to_time_period(self, team_id, time_period_type='M', time_period_num=1):
    # {{{
        """
        Calculates (GS-GC) of specific team from goals scored and conceded in particular time period played before.
        This feature should somehow aggregate information about team attack and defensive strength.
        :param time_period_num: int:
            Specifies particular number of time period (specified in param 'time_period_type') from which the goals
            characteristics should be included.
        :param time_period_type: str:
            Possible values are: 'W' : week(fixed to 7 days)
                                 'M' : month(fixed to 30 days)
                                 'Y' : year(fixed to 365 days)
                                 'S' : season(using self.SL_data)
                                 'L' : life(using self.LL_data)
        :param team_id: int:
            Specifies particular team
        :return: int:
        """
        if time_period_type not in ['W', 'M', 'Y', 'S', 'L']:
            time_period_type = 'M'
        if type(time_period_num) is not int or time_period_num == 0:
            time_period_num = 1

        if time_period_type in ['W', 'M', 'Y', 'S']:
            goals_scored = np.nan
            goals_conceded = np.nan
            if time_period_type in ['W', 'M', 'Y']:
                matches_containing_team = self.matches[(self.matches["HID"] == team_id) |
                                                       (self.matches["AID"] == team_id)].sort_index()
                if time_period_type == 'W':
                    time_period_num *= 7  # week fixed to 7 days
                elif time_period_type == 'M':
                    time_period_num *= 30  # month fixed to 30 days
                elif time_period_type == 'Y':
                    time_period_num *= 365  # year fixed to 365 days

                how_deep_to_past = np.datetime64(self.today) - np.timedelta64(time_period_num, 'D')
                matches_containing_team = matches_containing_team[(matches_containing_team['Date'] >= str(how_deep_to_past))
                                                                  & (matches_containing_team['Date'] < self.yesterday)]
                if not matches_containing_team.empty:
                    goals_conceded = matches_containing_team[matches_containing_team["HID"] == team_id]['ASC'].sum() + \
                                     matches_containing_team[matches_containing_team["AID"] == team_id]['HSC'].sum()
                    goals_scored = matches_containing_team[matches_containing_team["HID"] == team_id]['HSC'].sum() + \
                                   matches_containing_team[matches_containing_team["AID"] == team_id]['ASC'].sum()

            elif time_period_type == 'S':
                # It is assumed that team is already added in DataFrame self.LL_data
                matches_containing_team = self.SL_data.xs(team_id, level='second')[-1-time_period_num:-1]
                if not matches_containing_team.empty:
                    goals_conceded = matches_containing_team['SL_Goals_Conceded'].sum()
                    goals_scored = matches_containing_team['SL_Goals_Scored'].sum()

            return goals_scored - goals_conceded
        elif time_period_type == 'L':
            # It is assumed that team is already added in DataFrame self.LL_data
            return self.LL_data.loc[team_id, 'LL_Goals_Scored'] - self.LL_data.loc[team_id, 'LL_Goals_Conceded']
    # }}}

    def goals_ratio_to_num_matches(self, team_id, num_matches=1):
    # {{{
        """
        Calculates (GS/GC) of specific team from goals scored and conceded in particular number of matches played before.
        This feature should somehow aggregate information about team attack and defensive strength.
        :param team_id: int:
            Specifies particular team
        :param num_matches: int:
            Specifies particular number of matches from which the goals characteristics should be included.
        :return: int:
        """
        if type(num_matches) is not int or num_matches == 0:
            num_matches = 1
        # this is fastest selecting to compared with concat and append
        # %timeit matches[(matches["HID"] == team_id) | (matches["AID"] == team_id)].sort_index()
        # 1.21 ms ± 14.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        # %timeit pd.concat([matches[matches["HID"] == team_id], matches[matches["AID"] == team_id]]).sort_index()
        # 3.26 ms ± 62.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        # %timeit matches[matches["HID"]==team_id].append(matches[matches["AID"]==team_id]).sort_index()
        # 3.31 ms ± 75.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        matches_containing_team = self.matches[(self.matches["HID"] == team_id) |
                                               (self.matches["AID"] == team_id)].sort_index()[-1-num_matches:-1]

        goals_conceded, goals_scored = np.nan, np.nan
        if not matches_containing_team.empty:
            goals_conceded = matches_containing_team[matches_containing_team["HID"] == team_id]['ASC'].sum() + \
                             matches_containing_team[matches_containing_team["AID"] == team_id]['HSC'].sum()
            goals_scored = matches_containing_team[matches_containing_team["HID"] == team_id]['HSC'].sum() + \
                           matches_containing_team[matches_containing_team["AID"] == team_id]['ASC'].sum()

        return goals_scored / goals_conceded if goals_conceded != 0 else goals_scored / (goals_conceded + 1)
    # }}}

    def goals_ratio_to_time_period(self, team_id, time_period_type='M', time_period_num=1):
    # {{{
        """
        Calculates (GS/GC) of specific team from goals scored and conceded in particular time period played before.
        This feature should somehow aggregate information about team attack and defensive strength.
        :param time_period_num: int:
            Specifies particular number of time period (specified in param 'time_period_type') from which the goals
            characteristics should be included.
        :param time_period_type: str:
            Possible values are: 'W' : week(fixed to 7 days)
                                 'M' : month(fixed to 30 days)
                                 'Y' : year(fixed to 365 days)
                                 'S' : season(using self.SL_data)
                                 'L' : life(using self.LL_data)
        :param team_id: int:
            Specifies particular team
        :return: int:
        """
        if time_period_type not in ['W', 'M', 'Y', 'S', 'L']:
            time_period_type = 'M'
        if type(time_period_num) is not int or time_period_num == 0:
            time_period_num = 1

        if time_period_type in ['W', 'M', 'Y', 'S']:
            goals_scored = np.nan
            goals_conceded = np.nan
            if time_period_type in ['W', 'M', 'Y']:
                matches_containing_team = self.matches[(self.matches["HID"] == team_id) |
                                                       (self.matches["AID"] == team_id)].sort_index()
                if time_period_type == 'W':
                    time_period_num *= 7  # week fixed to 7 days
                elif time_period_type == 'M':
                    time_period_num *= 30  # month fixed to 30 days
                elif time_period_type == 'Y':
                    time_period_num *= 365  # year fixed to 365 days

                how_deep_to_past = np.datetime64(self.today) - np.timedelta64(time_period_num, 'D')
                matches_containing_team = matches_containing_team[
                    (matches_containing_team['Date'] >= str(how_deep_to_past))
                    & (matches_containing_team['Date'] < self.yesterday)]
                if not matches_containing_team.empty:
                    goals_conceded = matches_containing_team[matches_containing_team["HID"] == team_id]['ASC'].sum() + \
                                     matches_containing_team[matches_containing_team["AID"] == team_id]['HSC'].sum()
                    goals_scored = matches_containing_team[matches_containing_team["HID"] == team_id]['HSC'].sum() + \
                                   matches_containing_team[matches_containing_team["AID"] == team_id]['ASC'].sum()

            elif time_period_type == 'S':
                # It is assumed that team is already added in DataFrame self.LL_data
                matches_containing_team = self.SL_data.xs(team_id, level='second')[-1-time_period_num:-1]
                if not matches_containing_team.empty:
                    goals_conceded = matches_containing_team['SL_Goals_Conceded'].sum()
                    goals_scored = matches_containing_team['SL_Goals_Scored'].sum()

            return goals_scored / goals_conceded if goals_conceded != 0 else goals_scored / (goals_conceded + 1)
        elif time_period_type == 'L':
            # It is assumed that team is already added in DataFrame self.LL_data
            gs, gc = self.LL_data.loc[team_id, 'LL_Goals_Scored'], self.LL_data.loc[team_id, 'LL_Goals_Conceded']

            return gs / gc if gc != 0 else gs / (gc + 1)
    # }}}

    def goals_to_match_ratio(self,ID, number_of_matches,MatchID=None, goal_type='scored'):
    # {{{
        '''
        Parametrs:
            ID(int): ID of the team.
            number_of_matches(int): Number of matches to evaluate.
            MatchID(int): MatchID for the feature
            goal_type(str): 'scored'/'conceded'(None)
        Returns:
            float: (scored / conceded) goals / # matches
        '''
        team_matches = self.match_data.loc[ID]
        if not MatchID:
            last_number_of_matches = team_matches.tail(number_of_matches)
        if MatchID:
            match_date = self.matches.loc[MatchID].Date
            previous_matches = team_matches[team_matches.Date < match_date]
            last_number_of_matches = previous_matches.tail(number_of_matches)

        if goal_type == 'scored':
            return last_number_of_matches.M_Goals_Scored.sum() / number_of_matches
        else:
            return last_number_of_matches.M_Goals_Conceded.sum() / number_of_matches
    # }}}

    def goals_ratio(self, ID, oppo_ID, matches = 1, vs = False):
    # {{{
        '''
        Returns (goals_scored/(goals_scored  + goals_conceded)) of first team or this vs statistics
        Parametrs:
            oppo_ID(int): ID of the opponent.
            ID(int): team id
            matches(int): numbers of matches to past
            vs(bool): set against each other
        Returns:
            float or 2 floats
        '''
        matches_period =self.matches[(self.matches["HID"]==ID) | (self.matches["AID"]==ID)].sort_index()[-1-matches:-1]
        if vs:
            matches_period =matches_period[matches_period["HID"]==oppo_ID].append(matches_period[matches_period["AID"]==oppo_ID]).sort_index()[-1-matches:-1]
        goals_conceded, goals_scored = np.nan, np.nan

        if not matches_period.empty:
            goals_conceded = matches_period[matches_period["HID"]==ID]['ASC'].sum() + \
                             matches_period[matches_period["AID"]==ID]['HSC'].sum()
            goals_scored = matches_period[matches_period["HID"]==ID]['HSC'].sum() + \
                           matches_period[matches_period["AID"]==ID]['ASC'].sum()
        goals_ID =goals_scored/(goals_scored + goals_conceded)
        if vs:
            return (goals_ID, (1-goals_ID))
        else:
            return goals_ID
    # }}}

    def wins_ratio(self, ID, months = None, matches=None):
    # {{{
        '''
        Returns wins in time or match period
        Parameters:
            ID(int): team id
            months(int) = numbers of months
            matches(int) = numbers of matches to past
        Returns:
            int
        '''
        if months != None:
            months_period =self.matches[self.matches['Date'].isin(pd.date_range(end=self.today, periods=(months*30), freq='D')[::-1])]
            wins = np.nan
            if not months_period.empty:
                wins = months_period[months_period["HID"] == ID]["H"].sum() + \
                       months_period[months_period["AID"] == ID]["A"].sum()
            return wins

        else:
            matches_period =self.matches[(self.matches["HID"] == ID) | (self.matches["AID"] == ID)].sort_index()[-1-matches:-1]
            wins = np.nan
            if not matches_period.empty:
                wins = matches_period[matches_period["HID"] == ID]['H'].sum() + \
                       matches_period[matches_period["AID"] == ID]['A'].sum()
            return wins
    # }}}

    def home_advantage(self, ID, MatchID = None, method=None):
    # {{{
        '''
        Calculate the home advantage feature of the team. t.i. (# home_wins)/(# home_plays) - (#wins)/(#plays)
        That is home_win_r - win_r. (The advantage of playing home against the total win rate).

        Parameters:
            MatchID(int/None): The `MatchID` to calculate the feature for. If `== None` than it counts all the matches that were recorded
            method(str/None): (`'rate_surplus`/`'rate_ratio'`/None).
                `'rate_surplus'` ->  `home_win_r - win_r`
                `'rate_ratio'` -> `home_win_r / win_r`
                `'None'` / `'rate'` -> `home_win_r`
                `'all'` -> returns a tuple of (rate_surplus,rate_ratio,rate)
        '''
        team_matches = self.match_data.loc[ID]
        team_matches_home = team_matches[team_matches.Home == 1]
        # if calculating with all the match IDs that are currently recorded
        if not MatchID:
            home_win_r = (team_matches_home['M_Win'] + team_matches_home['M_Draw'] * .5).sum()/len(team_matches_home)
            if method in ['rate_surplus','rate_ratio','all']:
                win_r = (team_matches['M_Win'] + team_matches['M_Draw'] * .5).sum()/len(team_matches)
        # calculate for some arbitrary MatchID with more matches being recorded than only the ones berfore the MatchID (for correlation purposes)
        else:
            match_date = self.matches.loc[MatchID].Date
            previous_home_matches = team_matches[(team_matches.Date < match_date) & (team_matches.Home == 1)]
            previous_matches = team_matches[team_matches.Date < match_date]
            home_win_r = (previous_home_matches['M_Win'] + previous_home_matches['M_Draw'] * .5).sum()/len(previous_home_matches)
            if method in ['rate_surplus','rate_ratio','all']:
                win_r = (previous_matches['M_Win'] + previous_matches['M_Draw'] * .5).sum()/len(previous_matches)
        if method == 'rate_surplus':
            return home_win_r - win_r
        elif method == 'rate_ratio':
            return home_win_r / win_r # test only away_lose_r
        elif method == 'all':
            return (home_win_r - win_r, home_win_r/win_r,home_win_r)
        else:
            return home_win_r
    # }}}

    def away_disadvantage(self, ID, MatchID = None, method=None):
    # {{{
        '''
        Calculate the away disadvantage feature of the team. t.i. (# away_loses)/(# away_plays) - (# loses)/(#plays)`
        That is away_lose_r - lose_r. (The advantage of playing home against the total win rate).

        Parameters:
            MatchID(int/None): The `MatchID` to calculate the feature for. If `== None` than it counts all the matches that were recorded
            method(str/None): (`'rate_surplus`/`'rate_ratio'`/None).
                `'rate_surplus'` ->  `away_lose_r - lose_r`
                `'rate_ratio'` -> `away_lose_r / lose_r`
                `'None'` / `'rate'` -> `away_lose_r`
                `'all'` -> returns a tuple of (rate_surplus,rate_ratio,rate)
        '''
        team_matches = self.match_data.loc[ID]
        team_matches_away = team_matches[team_matches.Away == 1]
        # if calculating with all the match IDs that are currently recorded
        if not MatchID:
            away_lose_r = (team_matches_away['M_Lose'] + team_matches_away['M_Draw'] * .5).sum()/len(team_matches_away)
            if method in ['rate_surplus','rate_ratio','all']:
                lose_r = (team_matches['M_Lose'] + team_matches['M_Draw'] * .5).sum()/len(team_matches)
        # calculate for some arbitrary MatchID with more matches being recorded than only the ones berfore the MatchID (for correlation purposes)
        else:
            match_date = self.matches.loc[MatchID]['Date']
            previous_away_matches = team_matches[(team_matches.Date < match_date) & (team_matches.Away == 1)]
            previous_matches = team_matches[team_matches.Date < match_date]
            away_lose_r = (previous_away_matches['M_Lose'] + previous_away_matches['M_Draw'] * .5).sum()/len(previous_away_matches)
            if method in ['rate_surplus','rate_ratio','all']:
                lose_r = (previous_matches['M_Lose'] + previous_matches['M_Draw'] * .5).sum()/len(previous_matches)

        if method == 'rate_surplus':
            return away_lose_r - lose_r
        elif method == 'rate_ratio':
            return away_lose_r / lose_r # test only away_lose_r
        elif method == 'all':
            return (away_lose_r - lose_r,  away_lose_r / lose_r, away_lose_r)
        else:
            return away_lose_r
    # }}}

    def elo_diff(self, MatchID):
    # {{{
        '''
        Returns the difference of the ELO ratings of the two teams playing in the match. (ELO_home - ELO_away)
        '''
        return self.LL_data.loc[self.matches.loc[MatchID].HID].ELO_rating - self.LL_data.loc[self.matches.loc[MatchID].AID].ELO_rating
    # }}}

    ###################
    #  RETURN VALUES  #
    ###################

    def _UPDATE_features(self):
    # {{{
        '''
        Updates the features in the attribute `self.features`
        '''
        def update_for_match(row):
            MatchID = row.name
            match_date = self.matches.loc[MatchID].Date
            home_team,away_team  = self.matches.loc[MatchID].HID,self.matches.loc[MatchID].AID

            home_all_matches = self.match_data.loc[home_team]
            away_all_matches = self.match_data.loc[away_team]
            # only taking matches that are older, than the currently analysed
            if type(home_all_matches) == pd.DataFrame: # there are multiple matches
                home_matches = home_all_matches[self.match_data.loc[home_team].Date < match_date]
            elif type(home_all_matches) == pd.Series: # there is only one match therefore the `home_all_matches` is a `pd.Series`
                home_matches = home_all_matches if home_all_matches.loc['Date'] < match_date else None
            else: # there is no match
                home_matches = None


            if type(away_all_matches) == pd.DataFrame: # there are multiple matches
                away_matches = away_all_matches[self.match_data.loc[away_team].Date < match_date]
            elif type(home_all_matches) == pd.Series: # there is only one match therefore the `away_all_matches` is a `pd.Series`
                away_matches = away_all_matches if away_all_matches.loc['Date'] < match_date else None
            else: # there is no match
                away_matches = None

            if type(home_matches) == pd.DataFrame:
                home_last_15 = home_matches.tail(15)
                sum_home_last_15_r = home_last_15[['M_Goals_Scored','M_Goals_Conceded','M_Win','M_Lose','M_Draw']].sum()/len(home_last_15)
            elif type(home_matches) == pd.Series:
                sum_home_last_15_r = home_matches.loc[['M_Goals_Scored','M_Goals_Conceded','M_Win','M_Lose','M_Draw']]
            else:
                sum_home_last_15_r = None


            if type(away_matches) == pd.DataFrame:
                away_last_15 = away_matches.tail(15)
                sum_away_last_15_r = away_last_15[['M_Goals_Scored','M_Goals_Conceded','M_Win','M_Lose','M_Draw']].sum()/len(away_last_15)
            elif type(away_matches) == pd.Series:
                sum_away_last_15_r = away_matches.loc[['M_Goals_Scored','M_Goals_Conceded','M_Win','M_Lose','M_Draw']]
            else:
                sum_away_last_15_r = None



            if not sum_home_last_15_r is None:
                # H_GS_GC_diff_#15
                new_feature_frame.at[MatchID,'H_GS_GC_diff_#15'] = sum_home_last_15_r.M_Goals_Scored - sum_home_last_15_r.M_Goals_Conceded

                # H_GS_# && H_GC_# && H_WR_#
                new_feature_frame.at[MatchID, 'H_GS_#'] = sum_home_last_15_r.M_Goals_Scored
                new_feature_frame.at[MatchID, 'H_GC_#'] = sum_home_last_15_r.M_Goals_Conceded
                new_feature_frame.at[MatchID, 'H_WR_#'] = sum_home_last_15_r.M_Win
                new_feature_frame.at[MatchID, 'H_DR_#'] = sum_home_last_15_r.M_Draw
                new_feature_frame.at[MatchID, 'H_LR_#'] = sum_home_last_15_r.M_Lose
            else:
                # H_GS_GC_diff_#15
                new_feature_frame.at[MatchID,'H_GS_GC_diff_#15'] = np.nan

                # H_GS_# && H_GC_# && H_WR_#
                new_feature_frame.at[MatchID, 'H_GS_#'] = np.nan
                new_feature_frame.at[MatchID, 'H_GC_#'] = np.nan
                new_feature_frame.at[MatchID, 'H_WR_#'] = np.nan
                new_feature_frame.at[MatchID, 'H_DR_#'] = np.nan
                new_feature_frame.at[MatchID, 'H_LR_#'] = np.nan



            if not sum_away_last_15_r is None:
                # A_GS_GC_diff_#15
                new_feature_frame.at[MatchID,'A_GS_GC_diff_#15'] = sum_away_last_15_r.M_Goals_Scored - sum_away_last_15_r.M_Goals_Conceded

                # A_GS_# && A_GC_#
                new_feature_frame.at[MatchID, 'A_GS_#'] = sum_away_last_15_r.M_Goals_Scored
                new_feature_frame.at[MatchID, 'A_GC_#'] = sum_away_last_15_r.M_Goals_Conceded
                new_feature_frame.at[MatchID, 'A_WR_#'] = sum_away_last_15_r.M_Win
                new_feature_frame.at[MatchID, 'A_DR_#'] = sum_away_last_15_r.M_Draw
                new_feature_frame.at[MatchID, 'A_LR_#'] = sum_away_last_15_r.M_Lose
            else:
                # A_GS_GC_diff_#15
                new_feature_frame.at[MatchID,'A_GS_GC_diff_#15'] = np.nan

                # A_GS_# && A_GC_#
                new_feature_frame.at[MatchID, 'A_GS_#'] = np.nan
                new_feature_frame.at[MatchID, 'A_GC_#'] = np.nan
                new_feature_frame.at[MatchID, 'A_WR_#'] = np.nan
                new_feature_frame.at[MatchID, 'A_DR_#'] = np.nan
                new_feature_frame.at[MatchID, 'A_LR_#'] = np.nan

        # teams that do not have features yet
        unregistered_matches = np.setdiff1d(self.matches.index.to_numpy(), self.features.index.to_numpy())
        # new data frame that will be appended to self.features
        new_feature_frame = pd.DataFrame(columns=['H_GS_GC_diff_#15','A_GS_GC_diff_#15','H_GS_#','A_GS_#','H_GC_#','A_GC_#','H_WR_#','A_WR_#','H_DR_#','A_DR_#','H_LR_#','A_LR_#'], index=unregistered_matches)
        new_feature_frame.apply(update_for_match,axis=1)
        self.features = pd.concat((self.features,new_feature_frame)).sort_index()
    # }}}

    def return_values(self):
    # {{{
        '''
        Return the values of the features in `self.today`
        '''
        return self.features.loc[self.opps_matches]
    # }}}

# plain numpy runs it faster about 4 ms, njit not jit did nor give better performance (tested on np.ndarray with shape (74664, 2))
# plain numpy: 98.8 ms ± 189 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# using numba njit ( AKA jit(nopython=True)): 102 ms ± 122 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# using numba jit (AKA jit(nopython=False)): 102 ms ± 96.9 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
def fast(pairs):
    # {{{
    """
    Calculates sum of vals for specific team present in first column of param pairs
    :param pairs: np.ndarray:
        every row contains team index and in second column of row is value e.g.
        [[Team_ID, num_of_scored_goals] X (num_of_played_games * 2)]:
        this can represent pairs of team:num_of_scored_goals: [[5, 2], [8, 3], [3, 4], [10, 4], [10, 3], [3, 8]]
    :return:
    """
    teams = np.unique(pairs[:, 0])
    out = np.zeros((teams.size, 2))
    for i, team in enumerate(teams):
        num = pairs[pairs[:, 0] == team][:, 1].sum()
        out[i, 0], out[i, 1] = team, num
    return out
    # }}}
