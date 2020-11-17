# Feature Extracion

## class Data
Class for manipulating the data and extracting characteristics.

### Attributes
  - `self.today` (pd.datetime64): current date (`pandas._libs.tslibs.timestamps.Timestamp`)
  - `self.bankroll` (int): bankroll from the summary
  - `self.matches`

        index    || 'opps_Date'            | 'Sea'  | 'Date'       | 'Open'                      | 'LID'           | 'HID'        | 'AID'
        match ID || date of opps occurence | season | date of play | date of betting possibility | league ID (str) | home team ID | away team ID
                  | 'HSC'             | 'ASC'             | 'H'      | 'D'  | 'A'      | 'OddsH'          | 'OddsD'      | 'OddsA'
                  | home goals scored | away goals scored | home win | draw | away win | odds of home win | odds of draw | odds of away win
                  | 'P(H)'               | 'P(D)'           | 'P(A)'               | 'BetH'       | 'BetD'   | 'BetA'
                  | model prob. home win | model prob. draw | model prob. away win | bet home win | bet draw | bet away win

  - `self.team_index`

         LL: life-long
         index   || 'LID'            | 'LL_Goals_Scored' | 'LL_Goals_Conceded' | 'LL_Wins' | 'LL_Draws' | 'LL_Loses' | 'LL_Played'    | 'LL_Accu'
         team ID || league ID (list) | goals scored      | goals conceded      | wins      | draws      | loses      | played matches | model accuracy

  - `self.time_data`

         SL: season-long
         index          || 'SL_Goals_Scored' | 'SL_Goals_Conceded' | 'SL_Wins' | 'SL_Draws' | 'SL_Loses' | 'SL_Played'    | 'SL_Accu'
         season,team ID || goals scored      | goals conceded      | wins      | draws      | loses      | played matches | model accuracy

  - `self.match_data`

         index   || 'MatchID' | 'Date'       | 'Oppo'      | 'Side'       | 'M_Goals_Scored' | 'M_Goals_Conceded' | 'M_Win'   | 'M_Draw'
         team ID || match ID  | date of play | opponent id | side of play | goals scored     | goals concede      | match win | match draw
                  | 'M_Lose'   | 'M_P(Win)'      | 'M_P(Draw)'      | 'M_P(Lose)'      | 'M_Accu'
                  | match lose | model prob. win | model prob. draw | model prob. lose | model accuracy


# Important comments
All of this is written with the assumption that the working dir is the git root (`./pm_mode`).

## Interacting with the environment

1) importing
```python
from hackathon.src.environment import Environment
```

2) loading data
```python
dataset = pd.read_csv('./hackathon/data/training_data.csv', parse_dates=['Date', 'Open'])
```

3) initializing (without `model`)
```python
env = Environment(dataset, None)
```

4) some usefull functions examples
```python
# returns the same as `data.curr_inc`
env.get_incremental_data(pd.to_datetime('2001-12-30'))

# returns the same as `data.curr_opps`
env.get_opps(pd.to_datetime('2001-12-30'))

# returns the same as `data.curr_summary`
env.generate_summary(pd.to_datetime('2001-12-30'))
```

## Parsing dates
1) For testing it is extremely important to parse the data with dates...
```python
pd.read_csv('./hackathon/data/training_data.csv', parse_dates=['Date', 'Open'])
```
and not
```python
pd.read_csv('./hackathon/data/training_data.csv')
```
