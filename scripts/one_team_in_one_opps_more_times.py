#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from hackathon.src.environment import Environment

dataset = pd.read_csv('../hackathon/data/training_data.csv',parse_dates=['Open','Date'])

env = Environment(dataset,None)

for date in pd.date_range('2000-03-19','2011-01-01'):
    opps = env.get_opps(date)
    teams = np.concatenate((opps.AID.to_numpy(),opps.HID.to_numpy()))
    if teams.size != 0 | teams.size != 1:
        if any(np.sort(teams) != np.unique(teams)):
            print('same team twice in opp')
            break;

