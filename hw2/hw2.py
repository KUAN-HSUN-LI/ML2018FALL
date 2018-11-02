import sys
import numpy as np
import pandas as pd

input_X = sys.argv[1]
input_Y = sys.argv[2]

fileX = pd.read_csv(input_X)
fileY = pd.read_csv(input_Y)

w =[-1.6433349088765394, -0.2015613675553401, -0.06438319121973497, -3.491623087556751, -0.1486784086688451,
-0.07813713984258742, -0.17978626267968006, -1.1947029171068202, -1.7580897840846093, -0.4815103166790115, 
-1.079644110345111, -0.06431838191464648, -0.21834825967051327, 0.07224295496976277, 0.0032409098056563864, 
-0.3420982152085095, 0.13369135164441162, -0.5282865533415835, 0.44913327228371014, 1.7290896361441266, 
1.7647952640265276, 1.1828131835517806, 1.1439315342611733, 0.049458976703329105, 2.5572513239902115, 
-2.2239444123318246, -0.09565839262624881, -0.3222529032677735, -0.11237089994253836, -0.5380945361547593, 
0.016099933696173574, 0.0017018496828342293, -0.6439382866857902, 1.1001524278233978, 0.43208896861346474, 
1.2399939466048089, -0.3270642039237914, -0.2107210187247144, -0.13233108365422697, -3.193273693478143, 
0.209339017843321, 0.07668745452870142, -0.23089946654475518, -0.7007197383920221, 3.245953492395472, 
0.4445595382118097, -1.2787252515379093, -0.052552569505594955, -0.2393207783490165, -0.16142958097618829, 
8.771692019922472, 0.08770958270493012, -0.26010168439692366, -0.28513046130667524, -1.4994785943782563, 
-8.526688422381742, -0.10914610228472019, 1.4968265207386164, -0.09548227885844666, -0.17847271339830892, 
-0.17783866788393848, 0.13989327228816292, 0.06483051482506089, -0.48127480715327287, 1.5114237583107717, 
1.9691321409520168, 0.4594247106709593, 1.4968265207386164, 0.07008434681527939, -0.0772742490889832, 
-0.2727600018163151, 0.1328528703559002, 0.7489648891843098, 0.34662443961022443, -1.6443409591008937, 
0.7161724092864028, 0.09290987827721534, 11.79005294330319, 0.40560187384261664, 0.47125013164645907, 
0.31017103515657607, 0.17911204104977224, 0.2607072969825601, -0.05176756298210767, -3.6775202956474047, 
-4.610917723196231, -1.5325269313284755, -1.2651723273498618, -1.8855324865469234, -1.6747121554981745]
b = -0.14888536645081507

inputTest = sys.argv[3]
data = pd.read_csv(inputTest)
dataTestX = np.zeros((23, 10000))
for i in range(data.columns.size):
    dataTestX[i] = data[data.columns[i]]

# LIMIT_BAL(0)
limit_balT = np.zeros((1, 10000))
MEAN = np.mean(dataTestX[0])
STD = np.std(dataTestX[0])
MAX = np.max(dataTestX[0])
for i in range(10000):
    limit_balT[0][i] = dataTestX[0][i] / MAX

# SEX(1)
size = 10000
sexT = np.zeros((2, size))
for i in range(size):
    if dataTestX[1][i] == 1:
        sexT[0][i] = 0
        sexT[1][i] = 1
    else:
        sexT[0][i] = 1
        sexT[1][i] = 0

# EDUCATION(2) 有0 5 6的值，資料範圍1-4
educationT = np.zeros((7, size))
for i in range(size):
    for j in range(7):
        if dataTestX[2][i] == j:
            educationT[j][i] = 1

# MARRIAGE(3) 有0的值，資料範圍1-3
marriageT = np.zeros((4, size))
for i in range(size):
    for j in range(4):
        if dataTestX[3][i] == j:
            marriageT[j][i] = 1

# AGE(4)
ageT = np.zeros((1, size))
MEAN = np.mean(dataTestX[4])
STD = np.std(dataTestX[4])
MAX = np.max(dataTestX[4])
MIN = np.min(dataTestX[4])
for i in range(size):
    ageT[0][i] = (dataTestX[4][i] - MIN) / MAX

# PAY_0(5) [-2 - 8]
pay_0T = np.zeros((11, 10000))
for i in range(10000):
    for j in range(11):
        if dataTestX[5][i] == j - 2:
            pay_0T[j][i] = 1

# PAY_2(6) [-2 - 7]
pay_2T = np.zeros((10, 10000))
for i in range(10000):
    for j in range(10):
        if dataTestX[6][i] == j - 2:
            pay_2T[j][i] = 1

# PAY_3(7) [-2 - 8]
pay_3T = np.zeros((11, 10000))
for i in range(10000):
    for j in range(11):
        if dataTestX[7][i] == j - 2:
            pay_3T[j][i] = 1

# PAY_4(8) [-2 - 8]
pay_4T = np.zeros((11, 10000))
for i in range(10000):
    for j in range(11):
        if dataTestX[8][i] == j - 2:
            pay_4T[j][i] = 1

# pay_6(10) [-2 - 8] 1沒有
pay_5T = np.zeros((10, 10000))
for i in range(10000):
    for j in range(11):
        if j < 3:
            if dataTestX[10][i] == j - 2:
                pay_5T[j][i] = 1
        if j > 3:
            if dataTestX[10][i] == j - 2:
                pay_5T[j - 1][i] = 1

# pay_6(10) [-2 - 8] 1沒有
pay_6T = np.zeros((10, 10000))
for i in range(10000):
    for j in range(11):
        if j < 3:
            if dataTestX[10][i] == j - 2:
                pay_6T[j][i] = 1
        if j > 3:
            if dataTestX[10][i] == j - 2:
                pay_6T[j - 1][i] = 1

bill_and_payT = np.zeros((12, 10000))
for i in range(12):
    MEAN = np.mean(dataTestX[11 + i])
    STD = np.std(dataTestX[11 + i])
    MAX = np.max(dataTestX[11 + i])
    for j in range(10000):
        bill_and_payT[i][j] = dataTestX[11 + i][j] / MAX

xT = np.vstack((limit_balT, sexT, educationT, marriageT, ageT, pay_0T, pay_2T, pay_3T, pay_4T, pay_5T, pay_6T, bill_and_payT))

xTest = xT

yTest = np.zeros((10000))
yTest = 1/(1 + np.e**(-np.dot(w, xTest) - b))
count = 0
for i in range(10000):
    if(yTest[i] >= 0.5):
        count += 1
        yTest[i] = 1
    else:
        yTest[i] = 0
yTest = yTest.astype(int)

size = 10000
id = []
name = 'id_'
for i in range(size):
    id.append(name+str(i))

Output = sys.argv[4]

output = pd.DataFrame({'id':id, 'value':yTest})
output.to_csv(Output ,index=False,sep=',')