import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense

myname = 'Gagandeep'

# Preprocessing the data
lower_bound = 0.99
upper_bound = 1.01
seqorpar =28
arr = []
for layer in range(1,20):
    train_df = pd.read_csv('/kaggle/input/'+str(layer)+'_'+str(seqorpar)+'_poly_perf.csv',header=None)
    train_df = train_df.drop([0,2,3,4,5,6,7,8,9], axis=1)
    train_df = train_df.dropna(axis=1)
    npdata = np.array(train_df)
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
    elif arr[i][0] > arr[i][ver_2_GF]:
        out1.append(1)
        out2.append(0)
        out3.append(0)
    else:
        out1.append(0)
        out2.append(1)
        out3.append(0)

# print(arr[55])

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



#Training DNN Model

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(arr[:,:-3],arr[:,8:], epochs=10, batch_size=1)
# evaluate the keras model
_, accuracy = model.evaluate(arr[:,:-3],arr[:,8:])
print('Accuracy: %.2f' % (accuracy*100))


# Tournament Approach Calculating Number of winns.

for layer in range(1,20):
    arr_predict = []
    train_df = pd.read_csv('/kaggle/input/'+str(layer)+'_'+str(seqorpar)+'_poly_perf.csv',header=None)
    train_df = train_df.drop([0,1,2,3,4,5,6,7,8,9], axis=1)
    train_df = train_df.dropna(axis=1)
    npdata = np.array(train_df)
    for i in range(len(npdata)):
        for j in range(len(npdata)):
            arr_predict.append(np.concatenate((npdata[i],npdata[j]), axis=0))
    arr_predict = np.asarray(arr_predict)
    prediction = model.predict(arr_predict)
    f = open(myname+"result.txt", "a")
    f.write(str(layer)+'_'+str(seqorpar)+'_poly_perf.csv \n')
    f.close()
#     print(str(layer)+'_'+str(seqorpar)+'_poly_perf.csv')
#     print(prediction[:,0])
    version =0
    flag=0
    count_win =0
    for values in prediction[:,0]:
        if flag<len(npdata):
            flag +=1
            if values>0.5:
                count_win+=1
        else:
            f = open(myname+"result.txt", "a")
            f.write("Version "+str(version)+" : Win count = ")
            f.write("%.i \n" % count_win)
            f.close()
            version+=1
            count_win=0
            flag=0
