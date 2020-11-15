# Feature Extracion

## Attributes

> TODO: Copy some dovumentation <15-11-20, kunzaatko> >

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
