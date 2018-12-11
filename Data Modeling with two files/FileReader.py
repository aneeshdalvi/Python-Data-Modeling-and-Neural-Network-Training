import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import csv
import os


# data = pd.read_csv("data.csv")
local_path = "C:\\Users\\Aneesh Dalvi\\Desktop\\Fall 2018\\CSE 564\\data\\Output\\"

dataMap = pd.read_csv('C:\\Users\\Aneesh Dalvi\\Desktop\\Fall 2018\\CSE 564\\data\\Mapping.csv')
dataVc = pd.read_csv('C:\\Users\\Aneesh Dalvi\\Desktop\\Fall 2018\\CSE 564\\data\\VC.csv')
dataVh = pd.read_csv('C:\\Users\\Aneesh Dalvi\\Desktop\\Fall 2018\\CSE 564\\data\\Vh.csv')
# print(dataVh)
# file_length = len(data)
# print(data.Velocity)

# creating Sum function as the input contains string values also

with open(local_path + 'VC_Input.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['', 'Velocity', 'LanePos', 'SpeedLimit', 'Steer', 'Accel', 'Brake', 'LongAccel', 'LatAccel', 'HeadwayTime', 'HeadwayDist'])

with open(local_path + 'Vh_Output.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['', 'Mode', 'ResponseTime', 'Mistakes', 'Step'])


def Mean_List(A, start, interval):
    value = 0
    for j in range(start, start + interval):
        if(A[j] != '-'):
            value = value + float(A[j])

    if (interval == 0):
        return value
    return value / interval


# mappinf file columns
startIndex = dataMap.StartIndexVC
interval = dataMap.Interval

mappingLen = len(dataMap)

# print(interval)
# make Vc input file by taking average of all the values using mapping file
for i in range(len(interval)):
    mappingInterval = interval[i]
    mappingStartIndex = startIndex[i]

    velocity = 0
    lanePos = 0
    steer = 0
    speedLimit = 0
    accel = 0
    brake = 0
    longAccel = 0
    latAccel = 0
    headwayTime = 0
    headwayDist = 0
    # print(mappingStartIndex)
    velocity = Mean_List(dataVc.Velocity, mappingStartIndex, mappingInterval)
    lanePos = Mean_List(dataVc.LanePos, mappingStartIndex, mappingInterval)
    steer = Mean_List(dataVc.Steer, mappingStartIndex, mappingInterval)
    speedLimit = Mean_List(dataVc.SpeedLimit, mappingStartIndex, mappingInterval)
    accel = Mean_List(dataVc.Accel, mappingStartIndex, mappingInterval)
    brake = Mean_List(dataVc.Brake, mappingStartIndex, mappingInterval)
    longAccel = Mean_List(dataVc.LongAccel, mappingStartIndex, mappingInterval)
    latAccel = Mean_List(dataVc.LatAccel, mappingStartIndex, mappingInterval)
    headwayDist = Mean_List(dataVc.HeadwayDist, mappingStartIndex, mappingInterval)
    headwayDist = Mean_List(dataVc.HeadwayDist, mappingStartIndex, mappingInterval)
    # print(velocity)

    with open(local_path + 'VC_Input.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([i, velocity, lanePos, steer, speedLimit, accel, brake, longAccel, latAccel, headwayDist, headwayDist])

# creating Vh output file using mapping file
for i in range(mappingLen):
    vhIndex = dataMap.VHIndex[i]
    student = dataMap.Student[i]
    mode = dataMap.Mode[i]
    # print(vhIndex)
    if(vhIndex == -1):
        with open(local_path + 'Vh_Output.csv', 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([i, mode, '0', '0', '0'])
    else:
        with open(local_path + 'Vh_Output.csv', 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([i, mode, '0', dataVh.Mistakes[vhIndex], '0'])
