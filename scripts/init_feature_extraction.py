import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from hackathon.src.environment import Environment
from models.feature_extraction.feature_extraction import Data

if not 'limits' in locals():
    limits = {'start':'2000-03-20','end':'2011-06-30'}

dataset = pd.read_csv('./hackathon/data/training_data.csv', parse_dates=['Date', 'Open'])
env = Environment(dataset, None)
data = Data()
inc = [env.get_incremental_data(date) for date in pd.date_range(start=limits['start'], end=limits['end'])]
opps = [env.get_opps(date) for date in pd.date_range(start=limits['start'], end=limits['end'])]
summary = [env.generate_summary(date) for date in pd.date_range(start=limits['start'], end=limits['end'])]
