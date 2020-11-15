#!/usr/bin/env python

import pandas as pd
import numpy as np

dataset = pd.read_csv('../hackathon/data/training_data.csv', parse_dates=['Date','Open'])

seasons = dataset['Sea'].values

print([(ind,a) for (ind,a) in  enumerate(seasons == np.sort(seasons)) if a == False])

print('The seasons are sorted:', all(seasons == np.sort(seasons)))

year_season =  [(a.year, pd.to_datetime(str(b)).year) for (a,b) in dataset[['Date','Sea']].values]

print('The seasons follow the year:', all([(a == b) for (a,b) in year_season]))

