import numpy as np
import pandas as pd
import sys
from keras.models import load_model

test = sys.argv[1]
out = sys.argv[2]
data = pd.read_csv(test)

feature = np.array([row.split(" ") for row in data["feature"].tolist()],dtype=np.float32)
feature = np.reshape(feature, (np.shape(feature)[0] , 48,48,1))

def normalize(x):
    x = x/255.
    return x


m = load_model('model.h5')
xTest = normalize(feature)
yTest = m.predict(xTest)
output = np.argmax(yTest, axis = 1)
index = [i for i in range(0,output.size)]
outCsv = pd.DataFrame({'id' : index, 'label' : output})
outCsv.to_csv(out, index=0)