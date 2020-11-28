import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from models.feature_extraction.feature_extraction import Data

data = Data()
data.LL_data = pd.read_csv('./LL_data_final.csv',index_col=0)
data.matches = pd.read_csv('./matches_final.csv', parse_dates=['Date', 'Open'],index_col=0)
data.match_data = pd.read_csv('./match_data_final.csv', parse_dates=['Date'],index_col=0)
