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
    def __init__(self, sort_columns=True):
        '''
        Parameters:
            sort_columns(True): Sort the columns of the dataframes
        '''

        ########################
        #  private attributes  #
        ########################
        self._sort_columns = sort_columns
        self._curr_inc_teams= None # teams that are in inc
        self._curr_opps_teams= None # teams that are in opps

        ########################
        #  Storage attributes  #
        ########################
        self.yesterday = None # this is used for initialization in very first inc of data and then as reference to yesterday
        self.today = None # current date
        self.bankroll = None # current bankroll

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

        self.matches = pd.DataFrame(columns=['opps_Date','Sea','Date','Open','LID','HID','AID','HSC','ASC','H','D','A','OddsH','OddsD','OddsA','P(H)','P(D)', 'P(A)','BetH','BetD','BetA']) # All matches played by IDs ﭾ


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
            self._EVAL_summary(summary)

        if inc is not None:
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
            self.matches = self.matches[['opps_Date','Sea','Date','Open','LID','HID','AID','HSC','ASC','H','D','A','OddsH','OddsD','OddsA','P(H)','P(D)', 'P(A)','BetH','BetD','BetA']]
            self.LL_data = self.LL_data[['LID', 'LL_Goals_Scored','LL_Goals_Conceded','LL_Wins', 'LL_Draws', 'LL_Loses', 'LL_Played', 'LL_Accu']]

        # }}}

    def _EVAL_summary(self, summary):
        # {{{
        self.today = summary['Date'][0]
        self.bankroll = summary['Bankroll'][0]
        # }}}

    def _EVAL_inc(self, inc):
        # {{{
        self._eval_teams(inc, self._curr_inc_teams)
        self._eval_matches(inc)
        # }}}

    def _EVAL_opps(self, opps):
        # {{{
        self._eval_teams(opps, self._curr_inc_teams)
        self._eval_matches(opps)
        # }}}

    def _EVAL_P_dis(self, P_dis):
        # {{{
        self._eval_matches(P_dis)
        # }}}

    def _EVAL_bets(self, bets):
        # {{{
        self._eval_matches(bets)
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
                lids_frame = pd.concat((data_frame[['HID','LID']].set_index('HID'),data_frame[['AID','LID']].set_index('AID'))) # TODO: This will not work if there are multiple LIDs for one team in one inc <15-11-20, kunzaatko> # NOTE: This is probably working only because the inc already added some teams.
                lids = lids_frame[~lids_frame.index.duplicated(keep='first')].loc[index_new_teams]
                # Making a list from the 'LID's
                new_teams['LID'] = lids.apply(lambda row: np.array([row.LID]), axis=1) # this is costly but is only run once for each match %timeit dataset['LID'] = dataset.apply(lambda row: [row.LID], axis=1) -> 463 ms ± 13.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                self.LL_data = pd.concat((self.LL_data, new_teams))

            ##############
            #  NEW LIDS  #
            ##############

            # NOTE: This could be optimised radically but it has shown to be a pain in the ass so this is it. If there will be a 'TLE' (time limit exceeded) error, this is the place to change <15-11-20, kunzaatko> #

            # teams in the data_frame that are stored in the self.LL_data (teams that could have been changed)
            index_old_teams = np.intersect1d(index_self_teams,index_data_frame)
            index_old_teams_HID = np.intersect1d(index_old_teams, data_frame['HID'].to_numpy(dtype='int64'))
            index_old_teams_AID = np.intersect1d(index_old_teams, data_frame['AID'].to_numpy(dtype='int64'))

            for index in index_old_teams_HID:
                if not type(data_frame.set_index('HID').loc[index]) == pd.DataFrame:
                    if not data_frame.set_index('HID').loc[index]['LID'] in self.LL_data.at[index,'LID']:
                        self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])
                else:
                    if not data_frame.set_index('HID').loc[index].iloc[0]['LID'] in self.LL_data.at[index,'LID']:
                        self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('HID').at[index, 'LID'])

            for index in index_old_teams_AID:
                if not type(data_frame.set_index('AID').loc[index]) == pd.DataFrame:
                    if not data_frame.set_index('AID').loc[index]['LID'] in self.LL_data.at[index,'LID']:
                        self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])
                else:
                    if not data_frame.set_index('AID').loc[index].iloc[0]['LID'] in self.LL_data.at[index,'LID']:
                        self.LL_data.at[index,'LID'] = np.append(self.LL_data.at[index,'LID'],data_frame.set_index('AID').at[index, 'LID'])

            # see also (https://stackoverflow.com/questions/45062340/check-if-single-element-is-contained-in-numpy-array)}}}

    # TODO: Probably does not work correctly for the bets. The bets should not be combined for the `opps` and the `inc` but only for the `bets` dataframe. <17-11-20, kunzaatko> #
    # TODO: the 'opps_Date' is not working. The indexes should not be concatenated but appended for new matches if they do not have the same 'opps_Date'... (When they are not added on the same day) <17-11-20, kunzaatko> # -> the problem with this is though that we would have to groupby matchid to to access a match, and multiple MatchIDs would be the same in the dataframe -> We should consider adding a new frame with this data (or maybe the bets should be recorded as an associated series of the match... What is your oppinion/solution?
    def _eval_matches(self, data_frame):
        # {{{
        self.matches = self.matches.combine_first(data_frame)
        # }}}

        #####################################################################
        #  UPDATE THE FEATURES THAT CAN BE EXTRACTED FROM THE DATA IN SELF  #
        #####################################################################

    def update_features(self):
    # {{{
        '''
        Update the features for the data stored in `self`.
        '''
        self._UPDATE_LL_data_features()
        self._UPDATE_SL_data_features()
        self._UPDATE_match_data_features()
    # }}}

        self.yesterday = self.today # test me to not overwrite

    def _UPDATE_LL_data_features(self):
        '''
        TODO LL features are not cummulative, in very first iteration it is suddenly updated but it needs to be updated \
             already after very first match e.g. Match_ID=1 and this info used for training of model

        Populate all the features from the frame `self.LL_data`
        '''
        if self.yesterday is None:
            self.LL_data.fillna(0., inplace=True)
        # This is needed because some characteristics as score and who won is not present in matches_played at self.today
        matches_played_before = self.matches[self.matches['Date'] < self.today] if self.yesterday is None else \
            self.matches.groupby('Date').get_group(self.yesterday) if self.yesterday in self.matches['Date'].to_numpy() \
            else None

        self._update_LL_Played(matches_played_before)
        self._update_LL_Goals(matches_played_before)
        self._update_LL_Res(matches_played_before)
        self._update_LL_Accu(matches_played_before)

    def _update_LL_Played(self, matches_played):
        '''
        Update 'LL_Played' (games) of the fram self.LL_data
        :param matches_played: pd.Dataframe:
            Contains matches played at self.yesterday
        '''
        if matches_played is not None:
            teams_played = np.unique(np.concatenate((matches_played['HID'].to_numpy(dtype='int64'),
                                                     matches_played['AID'].to_numpy(dtype='int64'))), return_counts=True)
            self.LL_data['LL_Played'].loc[teams_played[0]] = self.LL_data['LL_Played'].loc[teams_played[0]] + teams_played[1]

    def _update_LL_Goals(self, matches_played):
        '''
        Update 'LL_Goals_Scored' and 'LL_Goals_Conceded' of the frame `self.LL_data`
        '''
        if matches_played is not None:
            teams_goals_scored = np.concatenate([matches_played[['HID', 'HSC']].to_numpy(dtype='int64'),
                                                 matches_played[['AID', 'ASC']].to_numpy(dtype='int64')])
            teams_goals_conceded = np.concatenate([matches_played[['HID', 'ASC']].to_numpy(dtype='int64'),
                                                   matches_played[['AID', 'HSC']].to_numpy(dtype='int64')])
            teams = np.unique(teams_goals_scored[:, 0])

            scored = fast(teams_goals_scored, teams)
            conceded = fast(teams_goals_conceded, teams)
            self.LL_data['LL_Goals_Scored'].loc[scored[:, 0]] = \
                self.LL_data['LL_Goals_Scored'].loc[scored[:, 0]] + scored[:, 1]
            self.LL_data['LL_Goals_Conceded'].loc[conceded[:, 0]] = \
                self.LL_data['LL_Goals_Conceded'].loc[conceded[:, 0]] + conceded[:, 1]

    def _update_LL_Res(self, matches_played):
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
            teams = np.unique(teams_wins[:, 0])

            wins = fast(teams_wins, teams)
            loses = fast(teams_loses, teams)
            draws = fast(teams_draws, teams)

            self.LL_data['LL_Wins'].loc[wins[:, 0]] = \
                self.LL_data['LL_Wins'].loc[wins[:, 0]] + wins[:, 1]
            self.LL_data['LL_Loses'].loc[loses[:, 0]] = \
                self.LL_data['LL_Loses'].loc[loses[:, 0]] + loses[:, 1]
            self.LL_data['LL_Draws'].loc[draws[:, 0]] = \
                self.LL_data['LL_Draws'].loc[draws[:, 0]] + draws[:, 1]

    def _update_LL_Accu(self, matches_played):
        '''
        Update 'LL_Accu' of the frame `self.LL_data`
        '''
        if matches_played is not None:
            pass

    def _UPDATE_SL_data_features(self):
    # {{{
        '''
        Populate all the features of `self.SL_data`
        '''
        # TODO: should be done incrementaly <17-11-20, kunzaatko> #

        if self.yesterday is None:
            self.LL_data.fillna(0., inplace=True)
        # This is needed because some characteristics as score and who won is not present in matches_played at self.today
        matches_played_before = self.matches[self.matches['Date'] < self.today] if self.yesterday is None else \
            self.matches.groupby('Date').get_group(self.yesterday) if self.yesterday in self.matches['Date'].to_numpy() \
            else None

        self._update_SL_Goals(matches_played_before)
        self._update_SL_Res(matches_played_before)
        self._update_SL_Played(matches_played_before)
        self._update_SL_Accu(matches_played_before)

    # TODO: Could be unified with `_update_LL_Goals` as `_update_Goals` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Goals(self, matches_played):
        '''
        Update 'SL_Wins', 'SL_Draws' and 'SL_Loses' of the frame `self.SL_data`
        '''
        pass
    # }}}

    # TODO: Could be unified with `_update_LL_Res` as `_update_Res` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Res(self, matches_played):
        pass

    # TODO: Could be unified with `_update_LL_Played` as `_update_Played` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Played(self, matches_played):
        pass

    # TODO: Could be unified with `_update_LL_Accu` as `_update_Accu` but for different frames. <17-11-20, kunzaatko> #
    def _update_SL_Accu(self, matches_played):
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
        if self.today in self.matches['Date'].values:
            # a dataframe of all the todays matches (matches that where played on `self.today`)
            matches_played_today = self.matches.groupby('Date').get_group(self.today)
            self._update_add_matches(matches_played_today)

        # TODO: should be done incrementaly <17-11-20, kunzaatko> #
    # }}}

    # FIXME: does not update the matches that are not gone through at today... The matches in the first inc. <18-11-20, kunzaatko> #
    def _update_add_matches(self, matches_played_today):
    # {{{
        '''
        Add the matches that were played today. The fields 'MatchID', 'Date' == self.today, 'Oppo' == HID/AID, 'Home' & 'Away' (int 1/0), 'M_Goals_Scored' & 'M_Goals_Conceded' (int), 'M_Win' & 'M_Draw' & 'M_Lose' (int 1/0), 'M_P(Win)' & 'M_P(Draw)' & 'M_P(Lose)' (float), 'M_Accu' should be filled.
        '''
        # the matches that played as home
        matches_home = matches_played_today.set_index('HID').drop(labels=['Open','opps_Date'],axis=1)
        renames = {'AID':'Oppo', 'HSC':'M_Goals_Scored', 'ASC':'M_Goals_Conceded', 'H':'M_Win', 'D':'M_Draw', 'A':'M_Lose', 'P(H)':'P(Win)', 'P(D)':'P(Draw)', 'P(A)':'P(Lose)'}
        matches_home.rename(renames, axis=1, inplace=True)
        matches_home['Home'] = 1
        matches_home['Away'] = 0
        matches_home['MatchID'] = matches_played_today.index
        # TODO: Model accuracy <17-11-20, kunzaatko> #

        # the matches that played as away
        matches_away = matches_played_today.set_index('AID').drop(labels=['Open','opps_Date'],axis=1)
        renames = {'HID':'Oppo', 'ASC':'M_Goals_Scored', 'HSC':'M_Goals_Conceded', 'A':'M_Win', 'D':'M_Draw', 'H':'M_Lose', 'P(A)':'P(Win)', 'P(D)':'P(Draw)', 'P(H)':'P(Lose)'}
        matches_away.rename(renames, axis=1, inplace=True)
        matches_away['Home'] = 0
        matches_away['Away'] = 1
        matches_away['MatchID'] = matches_played_today.index
        # TODO: Model accuracy <17-11-20, kunzaatko> #

        # TODO: Do not create a new object but only concat. <17-11-20, kunzaatko> #
        self.match_data = self.match_data.append([matches_away, matches_home])
    # }}}


    # ┌─────────────────────┐
    # │ MATCHES GLOBAL DATA │
    # └─────────────────────┘

    def matches_with(self, ID, oppo_ID):
        '''
        Returns all the matches with a particular opponent.

        Parameters:
            oppo_ID(int): ID of the opponent.
            ID(int): team id

        Returns:
            pd.DataFrame
        '''
        pass

    # ┌───────────────────────────┐
    # │ LIFE-LONG CHARACTERISTICS │
    # └───────────────────────────┘

    def total_scored_goals_to_match(self, ID, number_of_matches):
        '''
        Total life-long score to match ratio.

        Parameters:
            ID(int): team id
            number_of_matches(int): num

        Returns:
            float: scored goals / # matches
        '''
        pass

    def home_win_r(self):
        '''
        Win rate for win when home.

        Parameters:
            ID(int): team index

        Returns:
            float: rate of win, when home
        '''
        pass


@njit
def fast(pairs, teams):
    """
    ...
    :param pairs:
    :param teams:
    :return:
    """
    out = np.zeros((teams.size, 2))
    for i, team in enumerate(teams):
        num = pairs[pairs[:, 0] == team][:, 1].sum()
        out[i, 0], out[i, 1] = team, num
    return out