import pandas as pd
import numpy as np
data = pd.read_csv("train.csv",encoding='ISO-8859-15')
data_AMB = np.array([object()]*12)
data_CH4 = np.array([object()]*12)
data_CO = np.array([object()]*12)
data_NMHC = np.array([object()]*12)
data_NO = np.array([object()]*12)
data_NO2 = np.array([object()]*12)
data_NOx = np.array([object()]*12)
data_O3 = np.array([object()]*12)
data_PM10 = np.array([object()]*12)
data_PM2_5 = np.array([object()]*12)
#missing data
data_RH = np.array([object()]*12)
data_SO2 = np.array([object()]*12)
data_THC = np.array([object()]*12)
data_WD_HR = np.array([object()]*12)
data_WIND_DIREC = np.array([object()]*12)
data_WIND_SPEED = np.array([object()]*12)
data_WS_HR = np.array([object()]*12)

for i in range(12):
    data_AMB[i] = data.iloc[0+i*360,3::]
    data_AMB[i] = np.array(data_AMB[i], float)
    data_CH4[i] = data.iloc[1+i*360,3::]
    data_CH4[i] = np.array(data_CH4[i], float)
    data_CO[i] = data.iloc[2+i*360,3::]
    data_CO[i] = np.array(data_CO[i], float)
    data_NMHC[i] = data.iloc[3+i*360,3::]
    data_NMHC[i] = np.array(data_NMHC[i], float)
    data_NO[i] = data.iloc[4+i*360,3::]
    data_NO[i] = np.array(data_NO[i], float)
    data_NO2[i] = data.iloc[5+i*360,3::]
    data_NO2[i] = np.array(data_NO2[i], float)
    data_NOx[i] = data.iloc[6+i*360,3::]
    data_NOx[i] = np.array(data_NOx[i], float)
    data_O3[i] = data.iloc[7+i*360,3::]
    data_O3[i] = np.array(data_O3[i], float)
    data_PM10[i] = data.iloc[8+i*360,3::]
    data_PM10[i] = np.array(data_PM10[i], float)
    data_PM2_5[i] = data.iloc[9+i*360,3::]
    data_PM2_5[i] = np.array(data_PM2_5[i], float)
    #missing data
    data_RH[i] = data.iloc[11+i*360,3::]
    data_RH[i] = np.array(data_RH[i], float)
    data_SO2[i] = data.iloc[12+i*360,3::]
    data_SO2[i] = np.array(data_SO2[i], float)
    data_THC[i] = data.iloc[13+i*360,3::]
    data_THC[i] = np.array(data_THC[i], float)
    data_WD_HR[i] = data.iloc[13+i*360,3::]
    data_WD_HR[i] = np.array(data_WD_HR[i], float)    
    data_WIND_DIREC[i] = data.iloc[13+i*360,3::]
    data_WIND_DIREC[i] = np.array(data_WIND_DIREC[i], float)
    data_WIND_SPEED[i] = data.iloc[13+i*360,3::]
    data_WIND_SPEED[i] = np.array(data_WIND_SPEED[i], float)
    data_WS_HR[i] = data.iloc[13+i*360,3::]
    data_WS_HR[i] = np.array(data_WS_HR[i], float)
    
    
    for j in range(18+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_AMB[i] =np.hstack((data_AMB[i], temp))
    for j in range(19+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_CH4[i] =np.hstack((data_CH4[i], temp))        
    for j in range(20+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_CO[i] =np.hstack((data_CO[i], temp))
    for j in range(21+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_NMHC[i] =np.hstack((data_NMHC[i], temp))
    for j in range(22+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_NO[i] =np.hstack((data_NO[i], temp))
    for j in range(23+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_NO2[i] =np.hstack((data_NO2[i], temp))
    for j in range(24+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_NOx[i] =np.hstack((data_NOx[i], temp))
    for j in range(25+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_O3[i] =np.hstack((data_O3[i], temp))
    for j in range(26+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_PM10[i] =np.hstack((data_PM10[i], temp))
    for j in range(27+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_PM2_5[i] =np.hstack((data_PM2_5[i], temp))
    
    #missing data
    
    for j in range(29+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_RH[i] =np.hstack((data_RH[i], temp))
    for j in range(30+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_SO2[i] =np.hstack((data_SO2[i], temp))
    for j in range(20+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_THC[i] =np.hstack((data_THC[i], temp))
    for j in range(20+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_WD_HR[i] =np.hstack((data_WD_HR[i], temp))
    for j in range(20+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_WIND_DIREC[i] =np.hstack((data_WIND_DIREC[i], temp))
    for j in range(20+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_WIND_SPEED[i] =np.hstack((data_WIND_SPEED[i], temp))
    for j in range(20+i*360, 360*(i+1), 18):
        temp = data.iloc[j,3::]
        temp = np.array(temp, float)
        data_WS_HR[i] =np.hstack((data_WS_HR[i], temp))

for i in range(12):
    for j in range(data_AMB[i].size):
        if data_AMB[i][j] <= 0:
            data_AMB[i][j] = data_AMB[i][j-1]
        if data_CH4[i][j] <= 0:
            data_CH4[i][j] = data_CH4[i][j-1]
        if data_CO[i][j] <= 0:
            data_CO[i][j] = data_CO[i][j-1]
        if data_NMHC[i][j] <= 0:
            data_NMHC[i][j] = data_NMHC[i][j-1]
        if data_NO[i][j] <= 0:
            data_NO[i][j] = data_NO[i][j-1]
        if data_NO2[i][j] <= 0:
            data_NO2[i][j] = data_NO2[i][j-1]
        if data_NOx[i][j] <= 0:
            data_NOx[i][j] = data_NOx[i][j-1]
        if data_O3[i][j] <= 0:
            data_O3[i][j] = data_O3[i][j-1]
        if data_PM10[i][j] <= 0:
            data_PM10[i][j] = data_PM10[i][j-1]
        if data_PM2_5[i][j] <= 2 or data_PM2_5[i][j] > 120:
            data_PM2_5[i][j] = data_PM2_5[i][j-1]
        if data_RH[i][j] <= 0.0:
            data_RH[i][j] = data_RH[i][j-1]
        if data_SO2[i][j] <= 0:
            data_SO2[i][j] = data_SO2[i][j-1]
        if data_THC[i][j] < 0:
            data_THC[i][j] = data_THC[i][j-1]
        if data_WD_HR[i][j] < 0:
            data_WD_HR[i][j] = data_WD_HR[i][j-1]
        if data_WIND_DIREC[i][j] < 0:
            data_WIND_DIREC[i][j] = data_WIND_DIREC[i][j-1]
        if data_WIND_SPEED[i][j] < 0:
            data_WIND_SPEED[i][j] = data_WIND_SPEED[i][j-1]
        if data_WS_HR[i][j] < 0:
            data_WS_HR[i][j] = data_WS_HR[i][j-1]

#使用每筆data9小時內PM2.5的一次項（含bias項）
num = 9

lr = 0.1
iteration = 1000

month = 12



b = 0
w = [0]*num
b_lr = 0.0
w_lr = [0.0]*num

for i in range(iteration):
    b_grad = 0.0
    w_grad = [0.0]*num
    for j in range(month):
        for k in range(0, len(data_PM2_5[j])-9):
            data_past = [data_PM2_5[j][k], data_PM2_5[j][k+1], data_PM2_5[j][k+2], data_PM2_5[j][k+3], data_PM2_5[j][k+4], data_PM2_5[j][k+5], data_PM2_5[j][k+6], data_PM2_5[j][k+7], data_PM2_5[j][k+8]]
            formula = data_PM2_5[j][k+9] - b
            for l in range(num):
                formula -= data_past[l] * w[l]
            formula *= 2.0

            b_grad = b_grad - formula*1.0
            for l in range(num):
                w_grad[l] = w_grad[l] - formula * data_past[l]

        b_lr = b_lr + b_grad**2
        b = b - lr/np.sqrt(b_lr) * b_grad
        for k in range(num):
            w_lr[k] = w_lr[k] + w_grad[k]**2
            w[k] = w[k] - lr/np.sqrt(w_lr[k]) * w_grad[k]


error = 0
for j in range(month):
    for k in range(len(data_PM2_5[j]) - 9):
        estimation = b + data_PM2_5[j][k] * w[0] + data_PM2_5[j][k+1] * w[1] + data_PM2_5[j][k+2] * w[2] + data_PM2_5[j][k+3] * w[3] + data_PM2_5[j][k+4] * w[4] + data_PM2_5[j][k+5] * w[5] + data_PM2_5[j][k+6] * w[6] + data_PM2_5[j][k+7] * w[7] + data_PM2_5[j][k+8] * w[8]
        error += (np.abs(data_PM2_5[j][k + 9] - estimation))**2
error /= (471 * 12)
error = np.sqrt(error)