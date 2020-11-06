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
 
 ### Inputs
 Depends on extracted features and way of treating of time dimension.
 Can look something like:
 - X = [Team_ID, Opponent_ID, [Some features extracted]]
 - y = [H, D, A] e.g. [0, 0, 1] (if "Home" will be one feature of team and teams treated as 'team' and 'opponent' then this needs to be in correct order)
 ### Proposed features
 ##### Team specific
 | Name | Describtion | Range |
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
 
 
 ### Outputs
 - [Match_ID, P(H), P(D), P(A)] 
 - or based on  ([Team_ID, Opponent_ID, [Some features extracted]]) 'schema'
 it should be rather [Match_ID, P(Win_of_Team(Loss_of_opponent)), P(Draw), P(Loss_of_team(Win_of_opponent))]

 
 
 
 