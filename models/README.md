# Predictive models
Our task is to predict possible outcome of match as [P(H), P(D), P(A)].
## Subtasks:
1. Which features(inputs) are relevant to get best best predictions ?
2. What is relevant metric to evaluate results of model ?
3. How to deal with time dimension of data ?
4. Which model is the best in terms of speed and accuracy ?
### Proposed models to try:
 - ELO rating
 - Time Independent Poisson regression
 - Time dependent Poisson regression
 - Logistic regression
 - Random forest
 - another ensemble model (Gradient Boosting, Ada boosting, Voting, XGboosting etc.)

### Proposed features
##### Team specific
  | Name | Description | Range |
  | :---: | :---: | :---: |
  | Home | 1 if the team is home otherwise 0 | {0, 1} |
  | New | 1 if the team is new in league in the current season| {0, 1}|
  |"Some" team ranking as ELO etc. | - | - |
  |GS_last| goals scored last match | {0, 1 ...}|
  |GS_"something| goals scored in some time period (5 days, 20, season)| {0, 1, ...}|
  |GC_last| goals conceded last game| {0, 1 ...}|
  |GC_"something"| goals conceded in some time period | {0, 1, ...}|
  |GDiff_"something"|goals scored minus goals conceded in some time period| {..., -1, 0, 1, ...}|
  |MP|matches played in some time period|{0, 1, ...}|
  |AS|attack strength = ratio of the team’s average number of goals scored and the league’s average number of goals scored| R|
  |DS|defence strength = ratio of the team’s average number of goals allowed and the league’s average number of goals allowed| R|

## API structure

```python
class Model_XXX:
    def __init__(self, **model_params):
        self.P_dis = None
        ### CODE ###

    def run_iter(self, inc, opps):
        '''
        Run the iteration of the evaluation loop.

        Parameters:
        inc(pandas.DataFrame):  'DataFrame' with the played matches.
                                Includes 'HID', 'AID', 'LID', 'H', 'D' and 'A'.
        opps(pandas.DataFrame): 'DataFrame' with the betting opportunities.
                                Include 'MatchID', 'HID' and 'AID'.
        Returns:
        pandas.DataFrame: 'DataFrame' logging the process of `P_dis_get` under this model.
        '''
        ### CODE ###

        self.P_dis = P_dis
        return log
```

`self.P_dis` is a `pd.DataFrame(data=data, columns=['P(H)','P(D)','P(A)'], index=opps.index)`, neboli pravděpodobnosti událostí (pojmenování je podstatné) indexované pomocí `MatchID`.

 In `evaluation.py` it is important to rename the `model.py` in the same directory to anything else and add:

```python
import sys
sys.path.append(".")
```

in order to run the model located in the working directory of ipython REPL or Jupyter notebook.

The `Model_XXX` being used can be changed through a local variable `model` before the running of the `evaluation.py`. You can add `model_params` (`dict`) as a local variable that will be passed to the model being run. `bet_distribution` and `bet_distribution_params` can also be set as local variables.

### Examples:

__`Model_XXX.P_dis`__ (accessed by `model.model.P_dis`)
|      |     P(H) |   P(D) |     P(A) |\n|-----:|---------:|-------:|---------:|\n| 8142 | 0.413943 |      0 | 0.586057 |\n| 8143 | 0.413676 |      0 | 0.586324 |\n| 8144 | 0.698244 |
0 | 0.301756 |\n| 8145 | 0.3781   |      0 | 0.6219   |\n| 8147 |
0.604142 |      0 | 0.395858 |\n| 8150 | 0.308096 |      0 | 0.691904 |\n| 8169 | 0.477706 |      0 | 0.522294 |\n| 8170 | 0.233206
|      0 | 0.766794 |\n| 8174 | 0.25535  |      0 | 0.74465  |\n|
8175 | 0.558384 |      0 | 0.441616 |\n| 8176 | 0.59613  |      0
| 0.40387  |\n| 8177 | 0.566659 |      0 | 0.433341 |\n| 8178 | 0.632136 |      0 | 0.367864 |\n| 8179 | 0.170239 |      0 | 0.829761 |\n| 8180 | 0.333415 |      0 | 0.666585 |\n| 8181 | 0.270507 |
     0 | 0.729493 |\n| 8183 | 0.373879 |      0 | 0.626121 |\n| 8188 | 0.165287 |      0 | 0.834713 |\n| 8189 | 0.805155 |      0 |
0.194845 |\n| 8192 | 0.503741 |      0 | 0.496259 |\n| 8193 | 0.509331 |      0 | 0.490669 |

The log can be accessed by `model.log == (log_model, log_bet_distribution)`.

The `bet_distribution.bets` (accessed by `model.bet_distribution.bets`)
|      |   BetH |   BetD |   BetA |\n|-----:|-------:|-------:|-------:|\n| 8083 |      0 |      0 |      0 |\n| 8084 |
  0 |      0 |      0 |\n| 8085 |      0 |      0 |      0 |\n| 8086 |      0 |      0 |      0 |\n| 8087 |      0 |      0 |      0 |\n| 8094 |      0 |      0 |      0 |\n| 8096 |      0 |      0
|    100 |\n| 8102 |      0 |      0 |      0 |\n| 8103 |      0 |      0 |      0 |\n| 8118 |      0 |      0 |      0 |\n| 8119 |
     0 |      0 |      0 |\n| 8122 |      0 |      0 |      0 |\n| 8124 |      0 |      0 |      0 |\n| 8125 |      0 |      0 |
  0 |\n| 8128 |      0 |      0 |      0 |\n| 8129 |      0 |
 0 |      0 |

