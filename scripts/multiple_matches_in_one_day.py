#!/usr/bin/env python

import pandas as pd
import numpy as np

dataset = pd.read_csv('../hackathon/data/training_data.csv', parse_dates=['Date','Open'])

print('Searching for multiple matches of one team in one day:')
dates = dataset['Open'].to_numpy()
dataset = dataset.set_index('Open')

for date in dates:
    matches = dataset.loc[date]
    if not len(matches) > 1:
        ID = pd.concat((matches['AID'],matches['HID'])).to_numpy()
        if not all(ID == np.unique(ID)):
            print('    This team played multiple matches in one day:')
            print('        ', ID[np.unique(ID,return_counts=True)[1] > 1])



