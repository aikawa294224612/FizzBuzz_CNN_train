#!/usr/bin/env python
# coding: utf-8

# In[156]:


# [1,0,0,0] = 一般數字
# [0,1,0,0] = fizz
# [0,0,1,0] = buzz
# [0,0,0,1] = fizzbuzz

def fizzbuzz(n):   
    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return list([0,0,0,1])
    elif n % 3 == 0:
        return list([0,1,0,0])
    elif n % 5 == 0:
        return list([0,0,1,0])
    else:
        return list([1,0,0,0])


# In[162]:


def dec2bin(num):
    l = []    
    if num < 0:
        return '-' + dec2bin(abs(num))
    while True:
        num, remainder = divmod(num, 2)
        l.append(remainder)
        if num == 0:
            if len(l) < 12:
                add = []
                for i in range(12-len(l)):
                    add.append(0)
                add.extend(l[::-1])
                return add
            return l[::-1]
print(dec2bin(2000))


# In[172]:


import numpy as np

numlist_train = []
taglist_train = []
numlist_test = []
taglist_test = []

# train
for i in range(1,2000):
    numlist_train.append(dec2bin(i))
    taglist_train.append(fizzbuzz(i))    
# test
for i in range(2001,2048):
    numlist_test.append(dec2bin(i))
    taglist_test.append(fizzbuzz(i))

x_train = np.array(numlist_train)
y_train = np.array(taglist_train)

x_test = np.array(numlist_test)
y_test = np.array(taglist_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(y_test[0]) # 2001
print(y_test[1])  # 2002
print(y_test[2])  # 2003
print(y_test[3])  # 2004


# In[122]:


get_ipython().system('pip install matplotlib')


# In[184]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import matplotlib.pyplot as plt

dim_size = len(x_train[0])
tag_size = len(y_train[0])
dropoutrate = 0.4

model = Sequential()
model.add(Dense(input_dim = dim_size, units = 1000, activation='relu'))
model.add(Dropout(dropoutrate))
model.add(Dense(units= tag_size, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=20, epochs=100)

result= model.evaluate(x_test, y_test, batch_size = 1000)
print("test Acc:", result[1])


predicts = model.predict(x_test) 

p1, p2, p3, p4, p5 = predicts[0:5]

def getMax(pred):
    max_index = list(pred).index(max(list(pred)))   
    if max_index is 0:
        print("other")
    elif max_index is 1:
        print("fizz")
    elif max_index is 2:
        print("buzz")
    else:
        print("fizzbuzz")

getMax(p1)  # 2001  # fizz
getMax(p2)  # 2002  # other
getMax(p3)  # 2003  # other
getMax(p4)  # 2004  # fizz
getMax(p5)  # 2005  # buzz





