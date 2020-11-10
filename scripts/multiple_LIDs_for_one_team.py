#!/usr/bin/env python

import numpy as np
import pandas as pd

dataset = pd.read_csv('../hackathon/data/training_data.csv')

teams = dataset['HID'].unique()
teams.sort()

print("Printing teams that play in more than one league:")
for team in teams:
    LIDs = []
    for team_check, lid_check in zip(dataset['HID'],dataset['LID']):
        if team_check == team and lid_check not in LIDs:
            LIDs.append(lid_check)
    if len(LIDs) != 1:
        print(f"team {team} plays in leagues {LIDs}")
