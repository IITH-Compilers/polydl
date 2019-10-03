import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense

#Either run the file directly on kaggle or install the required modules and 
#give the proper location of the file

import os
for dirname, _, filenames in os.walk('/kaggle/input/10_1_poly_perf.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('/kaggle/input/10_1_poly_perf.csv',header=None)
train_df = train_df.drop([0,2,3,4,5,6,7,8,9], axis=1)
train_df = train_df.dropna(axis=1)
npdata = np.array(train_df)


# Preprocessing the data #################
lower_bound = 0.99
upper_bound = 1.01

arr = []
for i in range(len(npdata)):
    for j in range(len(npdata)):
        arr.append(np.concatenate((npdata[i],npdata[j]), axis=0))
arr = np.asarray(arr)

ver_2_GF = npdata.shape[1]


# 1 x>y
# 2 x<y
# 3 in range


out1 = []
out2 = []
out3 = []
out4 = []
for i in range(len(arr)):
    if (arr[i][0])*lower_bound <= arr[i][ver_2_GF] and (arr[i][0])*upper_bound >= arr[i][ver_2_GF]:
        out1.append(0)
        out2.append(0)
        out3.append(1)
        out4.append(3)
    elif arr[i][0] > arr[i][ver_2_GF]:
        out1.append(1)
        out2.append(0)
        out3.append(0)
        out4.append(1)
    else:
        out1.append(0)
        out2.append(1)
        out3.append(0)
        out4.append(2)


arr = pd.DataFrame(arr)
arr = arr.drop([0,ver_2_GF],axis=1)
arr["sum"] = arr.sum(axis=1)
arr = arr.loc[:,[1,2,3,4,6,7,8,9]].div(arr["sum"], axis=0)
arr.head()

out1 = pd.DataFrame(out1)
out2 = pd.DataFrame(out2)
out3 = pd.DataFrame(out3)

arr = pd.concat([arr, out1, out2, out3], axis=1, sort=False)
arr = np.array(arr)

# Applying Neural Network Model Here ###############

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# compile the keras model #################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(arr[:,:-3],arr[:,8:], epochs=5, batch_size=1)
# evaluate the keras model
_, accuracy = model.evaluate(arr[:,:-3],arr[:,8:])
print('Accuracy: %.2f' % (accuracy*100))