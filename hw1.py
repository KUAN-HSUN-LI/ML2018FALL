import sys
import pandas as pd
import numpy as np
Input = sys.argv[1]
Output = sys.argv[2]

data_test = pd.read_csv(Input,encoding='ISO-8859-15',names=['0','1','2','3','4','5','6','7','8','9'])

size = 260
hour = 9
data_test_AMB = np.array([object()]*260)
data_test_CH4 = np.array([object()]*260)
data_test_CO = np.array([object()]*260)
data_test_NMHC = np.array([object()]*260)
data_test_NO = np.array([object()]*260)
data_test_NO2 = np.array([object()]*260)
data_test_NOx = np.array([object()]*260)
data_test_O3 = np.array([object()]*260)
data_test_PM10 = np.array([object()]*260)
data_test_PM2_5 = np.array([object()]*260)
# missing data
data_test_RH = np.array([object()]*260)
data_test_SO2 = np.array([object()]*260)
data_test_THC = np.array([object()]*260)
data_test_WD_HR = np.array([object()]*260)
data_test_WIND_DIREC = np.array([object()]*260)
data_test_WIND_SPEED = np.array([object()]*260)
data_test_WS_HR = np.array([object()]*260)

for i in range(size):
    data_test_AMB[i] = data_test.iloc[0+i*18,1::]
    data_test_AMB[i] = np.array(data_test_AMB[i], float)
    data_test_CH4[i] = data_test.iloc[1+i*18,1::]
    data_test_CH4[i] = np.array(data_test_CH4[i], float)
    data_test_CO[i] = data_test.iloc[2+i*18,1::]
    data_test_CO[i] = np.array(data_test_CO[i], float)
    data_test_NMHC[i] = data_test.iloc[3+i*18,1::]
    data_test_NMHC[i] = np.array(data_test_NMHC[i], float)
    data_test_NO[i] = data_test.iloc[4+i*18,1::]
    data_test_NO[i] = np.array(data_test_NO[i], float)
    data_test_NO2[i] = data_test.iloc[5+i*18,1::]
    data_test_NO2[i] = np.array(data_test_NO2[i], float)
    data_test_NOx[i] = data_test.iloc[6+i*18,1::]
    data_test_NOx[i] = np.array(data_test_NOx[i], float)
    data_test_O3[i] = data_test.iloc[7+i*18,1::]
    data_test_O3[i] = np.array(data_test_O3[i], float)
    data_test_PM10[i] = data_test.iloc[8+i*18,1::]
    data_test_PM10[i] = np.array(data_test_PM10[i], float)
    data_test_PM2_5[i] = data_test.iloc[9+i*18,1::]
    data_test_PM2_5[i] = np.array(data_test_PM2_5[i], float)
    #missing data
    data_test_RH[i] = data_test.iloc[11+i*18,1::]
    data_test_RH[i] = np.array(data_test_RH[i], float)
    data_test_SO2[i] = data_test.iloc[12+i*18,1::]
    data_test_SO2[i] = np.array(data_test_SO2[i], float)
    data_test_THC[i] = data_test.iloc[13+i*18,1::]
    data_test_THC[i] = np.array(data_test_THC[i], float)
    data_test_WD_HR[i] = data_test.iloc[14+i*18,1::]
    data_test_WD_HR[i] = np.array(data_test_WD_HR[i], float)
    data_test_WIND_DIREC[i] = data_test.iloc[15+i*18,1::]
    data_test_WIND_DIREC[i] = np.array(data_test_WIND_DIREC[i], float)
    data_test_WIND_SPEED[i] = data_test.iloc[16+i*18,1::]
    data_test_WIND_SPEED[i] = np.array(data_test_WIND_SPEED[i], float)
    data_test_WS_HR[i] = data_test.iloc[17+i*18,1::]
    data_test_WS_HR[i] = np.array(data_test_WS_HR[i], float)

for i in range(size):
    for j in range(hour):
        if data_test_AMB[i][j] <= 0:
            data_test_AMB[i][j] = data_test_AMB[i][j-1]
        if data_test_CH4[i][j] <= 0:
            data_test_CH4[i][j] = data_test_CH4[i][j-1]
        if data_test_CO[i][j] <= 0:
            data_test_CO[i][j] = data_test_CO[i][j-1]
        if data_test_NMHC[i][j] <= 0:
            data_test_NMHC[i][j] = data_test_NMHC[i][j-1]
        if data_test_NO[i][j] <= 0:
            data_test_NO[i][j] = data_test_NO[i][j-1]
        if data_test_NO2[i][j] <= 0:
            data_test_NO2[i][j] = data_test_NO2[i][j-1]
        if data_test_NOx[i][j] <= 0:
            data_test_NOx[i][j] = data_test_NOx[i][j-1]
        if data_test_O3[i][j] <= 0:
            data_test_O3[i][j] = data_test_O3[i][j-1]
        if data_test_PM10[i][j] <= 0:
            data_test_PM10[i][j] = data_test_PM10[i][j-1]
        if data_test_PM2_5[i][j] <= 0 or data_test_PM2_5[i][j] > 120:
            data_test_PM2_5[i][j] = data_test_PM2_5[i][j-1]
        if data_test_RH[i][j] <= 0.0:
            data_test_RH[i][j] = data_test_RH[i][j-1]
        if data_test_SO2[i][j] <= 0:
            data_test_SO2[i][j] = data_test_SO2[i][j-1]
        if data_test_THC[i][j] < 0:
            data_test_THC[i][j] = data_test_THC[i][j-1]
        if data_test_WD_HR[i][j] < 0:
            data_test_WD_HR[i][j] = data_test_WD_HR[i][j-1]
        if data_test_WIND_DIREC[i][j] < 0:
            data_test_WIND_DIREC[i][j] = data_test_WIND_DIREC[i][j-1]
        if data_test_WIND_SPEED[i][j] < 0:
            data_test_WIND_SPEED[i][j] = data_test_WIND_SPEED[i][j-1]
        if data_test_WS_HR[i][j] < 0:
            data_test_WS_HR[i][j] = data_test_WS_HR[i][j-1]

w = [0.034575034499900965,
    -0.013926086615522457,
    0.11244860186384727,
    -0.1680460622856312,
    0.00032637232544317383,
    0.3836811174946772,
    -0.47779784624837013,
    -0.015295564241302472,
    1.1026186832217397]

b = 0.8869074904980341

ans = []
data_test = []
for i in range(size):
    y = b
    for j in range(hour):
        y += w[j] * data_test_PM2_5[i][j]
    ans.append(y)
    id = []
name = 'id_'
for i in range(260):
    id.append(name+str(i))

output = pd.DataFrame({'id':id, 'value':ans})
output.to_csv(Output,index=False,sep=',')
