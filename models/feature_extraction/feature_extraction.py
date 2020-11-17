import numpy as np
import pandas as pd

class Data:
    '''
    Class for manipulating the data and extracting characteristics.

    Attributes:
        today (pd.datetime64): current date (`pandas._libs.tslibs.timestamps.Timestamp`)
        self.bankroll (int): bankroll from the summary
    '''

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

        # `self.team_index`
        # LL: life-long
        # index   || 'LID'            | 'LL_Goals_Scored' | 'LL_Goals_Conceded' | 'LL_Wins' | 'LL_Draws' | 'LL_Loses' | 'LL_Played'    | 'LL_Accu'
        # team ID || league ID (list) | goals scored      | goals conceded      | wins      | draws      | loses      | played matches | model accuracy
        self.team_index = pd.DataFrame(columns=['LID','LL_Goals_Scored','LL_Goals_Conceded','LL_Wins', 'LL_Draws', 'LL_Loses', 'LL_Played', 'LL_Accu']) # recorded teams

        # `self.time_data`
        # SL: season-long
        # index          || 'SL_Goals_Scored' | 'SL_Goals_Conceded' | 'SL_Wins' | 'SL_Draws' | 'SL_Loses' | 'SL_Played'    | 'SL_Accu'
        # season,team ID || goals scored      | goals conceded      | wins      | draws      | loses      | played matches | model accuracy
        self.time_data = pd.DataFrame(columns=['SL_Goals_Scored', 'SL_Goals_Conceded', 'SL_Wins', 'SL_Draws', 'SL_Loses', 'SL_Played', 'SL_Accu']) # data frame for storing all the time characteristics for seasons


        # `self.match_data`
        # index   || 'MatchID' | 'Date'       | 'Oppo'      | 'Side'       | 'M_Goals_Scored' | 'M_Goals_Conceded' | 'M_Win'   | 'M_Draw'
        # team ID || match ID  | date of play | opponent id | side of play | goals scored     | goals concede      | match win | match draw
        #          | 'M_Lose'   | 'M_P(Win)'      | 'M_P(Draw)'      | 'M_P(Lose)'      | 'M_Accu'
        #          | match lose | model prob. win | model prob. draw | model prob. lose | model accuracy
        self.match_data = pd.DataFrame(columns=['MatchID', 'Date' , 'Oppo', 'Side',  'M_Goals_Scored', 'M_Goals_Conceded', 'M_Win','M_Draw', 'M_Lose','M_P(Win)','M_P(Draw)', 'M_P(Lose)','M_Accu'])


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
        # }}}

    def _eval_inc(self, inc):
        # {{{
        self._eval_teams(inc, self._curr_inc_teams)
        self._eval_matches(inc)
        # }}}

    def _eval_opps(self, opps):
        # {{{
        self._eval_teams(opps, self._curr_inc_teams)
        self._eval_matches(opps)
        # }}}

    def _eval_P_dis(self, P_dis):
        # {{{
        self._eval_matches(P_dis)
        # }}}

    def _eval_bets(self, bets):
        # {{{
        self._eval_matches(bets)
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

    def _eval_matches(self, data_frame):
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
        self._update_LL_time_features()
        self._update_time_data()
        self._update_season_time_features()
        self._update_last_match_features()

    def _update_time_data(self):
        '''
        Populate the `time_data` attribute from `matches`.
        '''
        # TODO: should be done incrementaly <17-11-20, kunzaatko> #
        pass

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
            float: scored goals /
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

