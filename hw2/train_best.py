import sys
import numpy as np
import pandas as pd

input_X = 'train_x.csv'
input_Y = 'train_y.csv'

fileX = pd.read_csv(input_X)
fileY = pd.read_csv(input_Y)

# training

dataY = np.zeros((20000))
for i in range(20000):
    dataY[i] = fileY[fileY.columns[0]][i]

dataX = np.zeros((23, 20000))
for i in range(fileX.columns.size):
    dataX[i] = fileX[fileX.columns[i]]

# LIMIT_BAL(0)
limit_bal = np.zeros((1, 20000))
MEAN = np.mean(dataX[0])
STD = np.std(dataX[0])
for i in range(20000):
    limit_bal[0][i] = (dataX[0][i] - MEAN) / STD 

# SEX(1)
sex = np.zeros((2, 20000))
for i in range(20000):
    if dataX[1][i] == 1:
        sex[0][i] = 0
        sex[1][i] = 1
    else:
        sex[0][i] = 1
        sex[1][i] = 0

# EDUCATION(2) 有0 5 6的值，資料範圍1-4
education = np.zeros((7, 20000))
for i in range(20000):
    for j in range(7):
        if dataX[2][i] == j:
            education[j][i] = 1

# MARRIAGE(3) 有0的值，資料範圍1-3
marriage = np.zeros((4, 20000))
for i in range(20000):
    for j in range(4):
        if dataX[3][i] == j:
            marriage[j][i] = 1

# AGE
age = np.zeros((1, 20000))
MEAN = np.mean(dataX[4])
STD = np.std(dataX[4])
for i in range(20000):
    age[0][i] = (dataX[4][i] - MEAN) / STD

# PAY_0(5) [-2 - 8]
pay_0 = np.zeros((11, 20000))
for i in range(20000):
    for j in range(11):
        if dataX[5][i] == j - 2:
            pay_0[j][i] = 1

# PAY_2(6) [-2 - 7]
pay_2 = np.zeros((10, 20000))
for i in range(20000):
    for j in range(10):
        if dataX[6][i] == j - 2:
            pay_2[j][i] = 1

# PAY_3(7) [-2 - 8]
pay_3 = np.zeros((11, 20000))
for i in range(20000):
    for j in range(11):
        if dataX[7][i] == j - 2:
            pay_3[j][i] = 1

# PAY_4(8) [-2 - 8]
pay_4 = np.zeros((11, 20000))
for i in range(20000):
    for j in range(11):
        if dataX[8][i] == j - 2:
            pay_4[j][i] = 1

# pay_6(10) [-2 - 8] 1沒有
pay_5 = np.zeros((10, 20000))
for i in range(20000):
    for j in range(11):
        if j < 3:
            if dataX[9][i] == j - 2:
                pay_5[j][i] = 1
        if j > 3:
            if dataX[9][i] == j - 2:
                pay_5[j - 1][i] = 1

# pay_6(10) [-2 - 8] 1沒有
pay_6 = np.zeros((10, 20000))
for i in range(20000):
    for j in range(11):
        if j < 3:
            if dataX[10][i] == j - 2:
                pay_6[j][i] = 1
        if j > 3:
            if dataX[10][i] == j - 2:
                pay_6[j - 1][i] = 1

bill_and_pay = np.zeros((12, 20000))
for i in range(12):
    MEAN = np.mean(dataX[11 + i])
    STD = np.std(dataX[11 + i])
    for j in range(20000):
        bill_and_pay[i][j] = (dataX[11 + i][j] - MEAN) / STD

x = np.vstack((limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, bill_and_pay))

xTrain = x

xSize = xTrain.shape[0]
ySize = 20000

lamb = 1

b = 1
w = np.ones((xSize))

lr = 0.1
iteration = 10001

b_lr = 1
w_lr = np.ones((xSize))
sigmoid = np.zeros((ySize))
b_grad = 0.0
w_grad = np.zeros((xSize))

Loss = 0

for i in range(iteration):
    
    sigmoid = 1/(1 + np.e**(-np.dot(w, xTrain) - b))
    formula = dataY - sigmoid
    
    b_grad = -np.sum(formula)
    w_grad = -np.dot(xTrain, formula)
    
    b_lr += b_grad**2
    w_lr += w_grad**2
    
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad