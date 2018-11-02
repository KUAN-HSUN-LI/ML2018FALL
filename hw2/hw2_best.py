import sys
import numpy as np
import pandas as pd

input_X = sys.argv[1]
input_Y = sys.argv[2]

fileX = pd.read_csv(input_X)
fileY = pd.read_csv(input_Y)

w =[-0.26247312193920147, -0.2271181830110357, -0.0898263320743135, -3.4572714466277437, -0.1606576961011506, -0.10796502768969177, 
-0.21787091490672386, -1.2241677494872527, -1.8035480370511054, -0.5448208604855712, -1.0960130294499528, 
-0.08688949399267626, -0.24697348333667027, 0.03217096030973923, 0.03221719092478699, -0.2787589294529807, 
0.15851830978686768, -0.522610924078256, 0.45396949005960213, 1.725181664623897, 1.7598752739231849, 
1.1256520972324866, 1.0945650827172388, -0.001871200443510478, 2.4853505886397596, -2.5569059556902762, 
-0.1135748278431131, -0.31910902602241653, -0.14778393134251547, -0.5003483505650198, -0.0389144810000171, 
-0.06705847711822477, -0.7009920063660009, 1.0107145240300397, 0.313556273032497, 1.2320600019430241, 
-0.35937527192254987, -0.17856419300947934, -0.1694244599127767, -3.386575180185636, 0.15793001366739337, 
0.0058508911001202485, -0.3214784984265061, -0.7511097373846568, 3.3926114712997326, 0.3274364066846469, 
-1.6224114814380362, -0.06931425968661496, -0.28505241916084123, -0.18263704328805944, 8.760975749490578, 
0.08252673424952976, -0.24789785249904298, -0.2567519832950412, -1.503049846885116, -8.849817068640979, 
-0.6407405901322399, 1.3345083202015637, -0.12686492255600046, -0.21091155896368627, -0.197334957666134, 
0.11697648154279723, 0.04172071187817252, -0.4903422378515843, 1.560642768666668, 2.961577706987248, 
1.1172870811657234, 1.3345083202015637, 0.05626597966831415, -0.09288571436504355, -0.2984746081280173, 
0.10680415449003472, 0.7127074563387329, 0.31068583153421975, -1.6696686723950906, 0.6824994746007583, 
-0.039605085287317356, 11.742697029822326, -0.09584188835446772, 0.10843349963880404, 0.20604996350067384, 
-0.0490754093740725, 0.11949576677059276, -0.11866567049173687, -0.18278483364898773, -0.30102732889873424, 
0.0035032022790033785, -0.02078668302961982, -0.03998462618462549, -0.051388803061491684]
b = -0.17462461667654663

inputTest = sys.argv[3]
data = pd.read_csv(inputTest)

dataTestX = np.zeros((23, 10000))
for i in range(data.columns.size):
    dataTestX[i] = data[data.columns[i]]

# LIMIT_BAL(0)
limit_balT = np.zeros((1, 10000))
MEAN = np.mean(dataTestX[0])
STD = np.std(dataTestX[0])
for i in range(10000):
    limit_balT[0][i] = (dataTestX[0][i] - MEAN) / STD

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
for i in range(size):
    ageT[0][i] = (dataTestX[4][i] - MEAN) / STD

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
    for j in range(10000):
        bill_and_payT[i][j] = (dataTestX[11 + i][j] - MEAN) / STD

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
