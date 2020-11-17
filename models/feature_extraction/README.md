# Feature Extracion

## class Data
Class for manipulating the data and extracting characteristics.

### Attributes

There are two types of public attributes:
1) __storage attributes__ - their purpose is simply to store data of the matches. To reiterate, we are not only storing the incremental data, that are in the input but the whole frames of data. (`self.today`, `self.bankroll`, `self.matches`)
2) __feature attributes__ - they store the data of the extracted features. The features themselves are meant to be extracted from the storage attributes (`self.LL_data`, `self.SL_data`, `self.match_data`)

  - `self.today` (`pandas._libs.tslibs.timestamps.Timestamp`): current date [__storage attribute__]
  - `self.bankroll` (`int`): bankroll from the summary [__storage attribute__]
  - `self.matches` (`pandas.DataFrame`) [__storage attribute__]

 | __index__ | `'opps_Date'`           | `'Sea'` | `'Date'`     | `'Open'`                    | `'LID'`         | `'HID'`      | `'AID'` | `'HSC'`           | `'ASC'`           | `'H'`    | `'D'` | `'A'`    | `'OddsH'`        | `'OddsD'`    | `'OddsA'` | `'P(H)'`             | `'P(D)'`         | `'P(A)'`             | `'BetH'`     | `'BetD'` | `'BetA'`                     |
 | :--       | :-:                     | :-:     | :-:          | :-:                         | :-:             | :-:          | :-:     | :-:               | :-:               | :-:      | :-:   | :-:      | :-:              | :-:          | :-:       | :-:                  | :-:              | :-:                  | :-:          | :-:      | :-:                          |
 | match ID  | date of opps occurrence | season  | date of play | date of betting possibility | league ID (str) | home team ID | away    | home goals scored | away goals scored | home win | draw  | away win | odds of home win | odds of draw | odds of   | model prob. home win | model prob. draw | model prob. away win | bet home win | bet draw | bet away win away winteam ID |


  - `self.LL_data` (`pandas.DataFrame`) [__feature attribute__]

    LL: life-long
 | __index__   | `'LID'`          | `'LL_Goals_Scored'` | `'LL_Goals_Conceded'` | `'LL_Wins'` | `'LL_Draws'` | `'LL_Loses'` | `'LL_Played'`  | `'LL_Accu'`    |
 | :-      | :-:              | :-:                 | :-:                   | :-:         | :-:          | :-:          | :-:            | :-:            |
 | team ID | league ID (list) | goals scored        | goals conceded        | wins        | draws        | loses        | played matches | model accuracy |


  - `self.SL_data` (`pandas.DataFrame`) [__feature attribute__]

    SL: season-long
 | __index (multi index)__ | `'LID'`          | `'SL_Goals_Scored'` | `'SL_Goals_Conceded'` | `'SL_Wins'` | `'SL_Draws'` | `'SL_Loses'` | `'SL_Played'`  | `'SL_Accu'`    |
 | :-                 | :-:              | :-:                 | :-:                   | :-:         | :-:          | :-:          | :-:            | :-:            |
 | season,team ID     | league ID (list) | goals scored        | goals conceded        | wins        | draws        | loses        | played matches | model accuracy |

  - `self.match_data` (`pandas.DataFrame`) [__feature attribute__]

 | __index__ | `'MatchID'` | `'Sea'` | `'Date'`     | `'Oppo'`    | `'Home'`     | `'Away'`     | `'M_Goals_Scored'` | `'M_Goals_Conceded'` | `'M_Win'` | `'M_Draw'` | `'M_Lose'` | `'M_P(Win)'`    | `'M_P(Draw)'`    | `'M_P(Lose)'`    | `'M_Accu'`     |
 | :-        | :-:         | :-:     | :-:          | :-:         | :-:          | :-:          | :-:                | :-:                  | :-:       | :-:        | :-:        | :-:             | :-:              | :-:              | :-:            |
 | team ID   | match ID    | season  | date of play | opponent id | team is home | team is away | goals scored       | goals conceded       | match win | match draw | match lose | model prob. win | model prob. draw | model prob. lose | model accuracy |



### Methods

a) __public__

  1)

  ```python
  def update_data(self, opps=None ,summary=None, inc=None, P_dis=None, bets=None):
      '''
      Run the iteration update of the data stored.
      ! Summary has to be updated first to get the right date!

      Parameters:
      All the parameters are supplied by the evaluation loop.
      opps(pandas.DataFrame): dataframe that includes the opportunities for betting.
      summary(pandas.DataFrame): includes the `Max_bet`, `Min_bet` and `Bankroll`.
      inc(pandas.DataFrame): includes the played matches with the scores for the model.
      '''
  ```

    Used every time that you want to incrementally update the _storage attributes_.

    2)

  ```python
  def update_features(self):
      '''
      Update the features for the data stored in `self`.
      '''
  ```

    Used every time that the features are to be evaluated from the data in the _storage attributes_ and updated in the _feature attributes_

b) __private__

1) The evaluation of the passed inputs is handled by the `self._EVAL_***` (where `***` is the input to evaluate) methods using the `self._eval_***` methods (where `***` is generally the field to update).
2) The updating of features from self is handled by the `self._UPDATE_***` (where `***` is the _feature attribute_ to update) methods using the `self._update_***` methods (where `***` is generally the field to update).






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
