#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from hackathon.src.environment import Environment

dataset = pd.read_csv('../hackathon/data/training_data.csv', parse_dates=['Date', 'Open'])

env = Environment(dataset, None)

print('Team multiple times in opps/inc:')
for date in pd.date_range(start='2000-03-20', end='2011-06-30'):
    inc = env.get_incremental_data(date)
    opps = env.get_opps(date)

    inc_HID,inc_AID = inc['HID'].to_numpy(dtype='int64'), inc['AID'].to_numpy(dtype='int64')
    inc_ID = np.concatenate((inc_AID,inc_HID))

    inc_ID_unique,inc_ID_unique_counts = np.unique(inc_ID, return_counts=True)
    inc_multi_playing_teams = inc_ID_unique[inc_ID_unique_counts != 1]
    if inc_multi_playing_teams.size > 0:
        print(f'inc({date}): {inc_multi_playing_teams}')

    opps_HID, opps_AID = opps['HID'].to_numpy(dtype='int64'), opps['AID'].to_numpy(dtype='int64')
    opps_ID = np.concatenate((opps_AID,opps_HID))

    opps_ID_unique,opps_ID_unique_counts = np.unique(opps_ID, return_counts=True)
    opps_multi_playing_teams = opps_ID_unique[opps_ID_unique_counts != 1]
    if opps_multi_playing_teams.size > 0:
        print(f'opps({date}): {opps_multi_playing_teams}')


