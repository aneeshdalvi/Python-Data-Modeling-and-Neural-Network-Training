

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import csv

#data = pd.read_csv("data.csv")
local_path = "C:\\Users\\Aneesh Dalvi\\Desktop\\564\\Assignment 2\\"

data = pd.read_csv('data.csv')
file_length = len(data)
# rint(data.Velocity)

header = ['Velocity', 'LanePos', 'SpeedLimit', 'Steer', 'Accel', 'Brake', 'LongAccel', 'HeadwayTime', 'HeadwayDist']
data.to_csv('Vc_output.csv', columns=header, index=False)

header = ['Mode', 'Response Time', 'Mistakes', 'Steps']
data.to_csv('Vh_output.csv', columns=header, index=False)
