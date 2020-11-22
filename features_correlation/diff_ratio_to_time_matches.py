import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from environment import Environment
from models.feature_extraction.feature_extraction import Data

limits = {'start': '2000-03-20', 'end': '2011-06-30'}

dataset = pd.read_csv('/home/emanuel/Documents/pm_mode/pm_mode/training_data.csv', parse_dates=['Date', 'Open'])
env = Environment(dataset, None)
data = Data()
inc = [env.get_incremental_data(date) for date in pd.date_range(start=limits['start'], end=limits['end'])]
opps = [env.get_opps(date) for date in pd.date_range(start=limits['start'], end=limits['end'])]
summary = [env.generate_summary(date) for date in pd.date_range(start=limits['start'], end=limits['end'])]


cols_diff_matches = ['HID', 'AID', 'H_diff_matches_1', 'H_diff_matches_2', 'H_diff_matches_5', 'H_diff_matches_10', 'H_diff_matches_15',
                     'H_diff_matches_25', 'HSC', 'ASC', 'H', 'D', 'A']
cols_diff_time = ['HID', 'AID', 'H_diff_week_1', 'H_diff_week_2', 'H_diff_month_1', 'H_diff_month_2',
                  'H_diff_month_5', 'H_diff_month_10', 'H_diff_month_15', 'H_diff_month_20', 'H_diff_year_1',
                  'H_diff_year_2', 'H_diff_year_5', 'H_diff_life',
                   # 'H_diff_sea_1', 'H_diff_sea_2', 'H_diff_sea_3',
                   'HSC', 'ASC', 'H', 'D', 'A']  # it is assumed that results for away would be similar

cols_ratio_matches = ['HID', 'AID', 'H_rat_matches_1', 'H_rat_matches_2', 'H_rat_matches_5', 'H_rat_matches_10', 'H_rat_matches_15',
                     'H_rat_matches_25', 'HSC', 'ASC', 'H', 'D', 'A']
cols_ratio_time = ['HID', 'AID', 'H_rat_week_1', 'H_rat_week_2', 'H_rat_month_1', 'H_rat_month_2',
                  'H_rat_month_5', 'H_rat_month_10', 'H_rat_month_15', 'H_rat_month_20', 'H_rat_year_1',
                  'H_rat_year_2', 'H_rat_year_5', 'H_rat_life',
                   # 'H_rat_sea_1', 'H_rat_sea_2', 'H_rat_sea_3',
                  'HSC', 'ASC', 'H', 'D', 'A']


table = pd.DataFrame(columns=cols_diff_time)
#table = pd.read_csv('/home/emanuel/Documents/pm_mode/pm_mode/models/feature_extraction/diff_time')
"""for o, i, s in zip(opps, inc, summary):
    data.update_data(opps=o, inc=i, summary=s)
    for i, row in i.iterrows():
        #  gs_gc difference related to number of matches
        '''data_ = [row.HID, row.AID, data.goals_difference_to_num_matches(row.HID),
                 data.goals_difference_to_num_matches(row.HID, num_matches=2),
                 data.goals_difference_to_num_matches(row.HID, num_matches=5),
                 data.goals_difference_to_num_matches(row.HID, num_matches=10),
                 data.goals_difference_to_num_matches(row.HID, num_matches=15),
                 data.goals_difference_to_num_matches(row.HID, num_matches=25),
                 row.HSC, row.ASC, row.H, row.D, row.A]'''
        #  gs_gc difference related to time period
        data_ = [row.HID, row.AID, data.goals_difference_to_time_period(row.HID, time_period_type='W'),
                 data.goals_difference_to_time_period(row.HID, time_period_type='W', time_period_num=2),
                 data.goals_difference_to_time_period(row.HID, time_period_type='M'),
                 data.goals_difference_to_time_period(row.HID, time_period_type='M', time_period_num=2),
                 data.goals_difference_to_time_period(row.HID, time_period_type='M', time_period_num=5),
                 data.goals_difference_to_time_period(row.HID, time_period_type='M', time_period_num=10),
                 data.goals_difference_to_time_period(row.HID, time_period_type='M', time_period_num=15),
                 data.goals_difference_to_time_period(row.HID, time_period_type='M', time_period_num=20),
                 data.goals_difference_to_time_period(row.HID, time_period_type='Y'),
                 data.goals_difference_to_time_period(row.HID, time_period_type='Y', time_period_num=2),
                 data.goals_difference_to_time_period(row.HID, time_period_type='Y', time_period_num=5),
                 data.goals_difference_to_time_period(row.HID, time_period_type='L'),
                 row.HSC, row.ASC, row.H, row.D, row.A]

        #  gs_gc ratio related to number of matches
        '''data_ = [row.HID, row.AID, data.goals_ratio_to_num_matches(row.HID),
                 data.goals_ratio_to_num_matches(row.HID, num_matches=2),
                 data.goals_ratio_to_num_matches(row.HID, num_matches=5),
                 data.goals_ratio_to_num_matches(row.HID, num_matches=10),
                 data.goals_ratio_to_num_matches(row.HID, num_matches=15),
                 data.goals_ratio_to_num_matches(row.HID, num_matches=25),
                 row.HSC, row.ASC, row.H, row.D, row.A]'''

        #  gs_gc ratio related to time period
        '''data_ = [row.HID, row.AID, data.goals_difference_to_time_period(row.HID, time_period_type='W'),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='W', time_period_num=2),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='M'),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='M', time_period_num=2),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='M', time_period_num=5),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='M', time_period_num=10),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='M', time_period_num=15),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='M', time_period_num=20),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='Y'),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='Y', time_period_num=2),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='Y', time_period_num=5),
                 data.goals_ratio_to_time_period(row.HID, time_period_type='L'),
                 row.HSC, row.ASC, row.H, row.D, row.A]'''

        other = pd.DataFrame(columns=cols_diff_time, data=[data_])
        table = table.append(other, ignore_index=True)
    data.update_features()


    print(s['Date'])
#table.to_csv(path_or_buf='diff_matches')
table.to_csv(path_or_buf='diff_time')"""
corrMatrix = table.corr()
fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sn.heatmap(corrMatrix, annot=True, linewidths=.7, ax=ax)
plt.show()
