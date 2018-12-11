import pandas as pd
import os
import csv


# creating Sum function as the input contains string values also
def Sum_List(A, i, offset):
    value = 0
    for j in range(i * offset, i * offset + len(A)):
        if(A[j] != '-'):
            value = value + float(A[j])
    return value


dataLocation = "C:\\Users\\Aneesh Dalvi\\Desktop\\Fall 2018\\CSE 564\\data\\data\\"
Vh_Output_Data = dataLocation + "Vh_Output_Data.xlsx"
local_path = "C:\\Users\\Aneesh Dalvi\\Desktop\\Fall 2018\\CSE 564\\data\\data\\answer\\"


flag = 0
current_student_id = ""
max_student = 30

# creating the output csv
with open(local_path + 'student_club.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["Velocity", 'LanePos', 'SpeedLimit', 'Steer', 'Accel', 'Brake', 'LongAccel', 'HeadwayTime', 'HeadwayDist', 'User', 'Mode', 'Speed', 'Number Of Errors', 'Response Time', 'Number of Steps'])


for folder in range(max_student):
    Student_Id = 1 + folder
    for Mode_Id in range(1, 4):
        #Mode_Id = Mode_Id + folder;
        folderName = dataLocation + "mode" + str(Mode_Id) + "\\Student" + str(Student_Id) + "mode" + str(Mode_Id) + "\\"
        # find the biggest file in this folder

        biggest = ("", -1)
        for item in os.listdir(folderName):
            itemsize = os.path.getsize(folderName + item)
            if itemsize > biggest[1] and item.endswith(".txt"):
                biggest = (folderName + item, itemsize)

        xls = pd.ExcelFile(Vh_Output_Data)
        df = xls.parse('Sheet1', index_col=None, na_values=['NA'])
        divide_offset = 0
        for k in range(flag + 1, len(df)):
            if(df.loc[flag][0] == Mode_Id):
                divide_offset = divide_offset + 1
                flag = flag + 1
            else:
                break

        data = pd.read_table(biggest[0],
                             usecols=['Velocity', 'LanePos', 'SpeedLimit', 'Steer', 'Accel', 'Brake', 'LongAccel', 'HeadwayTime', 'HeadwayDist'])

        file_length = len(data)

        # PAD ZEROS HERE
        if(file_length % divide_offset != 0):
            zero_lines_add = divide_offset - file_length % divide_offset

            # Adding zeros to start
            for loop in range(zero_lines_add // 2):
                data.loc[-1] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                data.index = data.index + 1
                data = data.sort_index()

            # Adding zeros to end
            zero_lines_add = zero_lines_add - zero_lines_add // 2
            for loop in range(zero_lines_add):
                data.loc[len(data)] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                data = data.sort_index()

        file_length = len(data)
        offset = file_length // divide_offset

        for i in range(divide_offset):

            # Fetching data from output file
            # Hardcoding taking first 6 lines from output VH fle

            xls = pd.ExcelFile(Vh_Output_Data)
            df = xls.parse('Sheet1', skiprows=0, index_col=None, na_values=['NA'])
            Mode = df.loc[i][0]
            Speed = df.loc[i][1]
            NOE = df.loc[i][2]
            Response_Time = df.loc[i][3]
            No_Steps = df.loc[i][4]

            df = data[i * offset: (i + 1) * offset]
            Velocity = Sum_List(df.Velocity, i, offset) / offset
            LanePos = Sum_List(df.LanePos, i, offset) / offset
            SpeedLimit = Sum_List(df.SpeedLimit, i, offset) / offset
            Steer = Sum_List(df.Steer, i, offset) / offset

            Accel = Sum_List(df.Accel, i, offset) / offset
            Brake = Sum_List(df.Brake, i, offset) / offset
            LongAccel = Sum_List(df.LongAccel, i, offset) / offset
            HeadwayTime = Sum_List(df.HeadwayTime, i, offset) / offset
            HeadwayDist = Sum_List(df.HeadwayDist, i, offset) / offset

            with open(local_path + 'student_club.csv', 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([Velocity, LanePos, SpeedLimit, Steer, Accel, Brake, LongAccel, HeadwayTime, HeadwayDist, Student_Id, Mode_Id, Speed, NOE, Response_Time, No_Steps])
