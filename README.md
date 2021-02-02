# FizzBuzz_CNN_train
Players generally sit in a circle. The player designated to go first says the number "1", and the players then count upwards in turn. However, any number divisible by three is replaced by the word fizz and any number divisible by five by the word buzz. Numbers divisible by 15 become fizz buzz. A player who hesitates or makes a mistake is eliminated from the game. (Wiki)

```python
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
```


```python
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
```

    [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    


```python
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
```

    (1999, 12)
    (1999, 4)
    (47, 12)
    (47, 4)
    [0 1 0 0]
    [1 0 0 0]
    [1 0 0 0]
    [0 1 0 0]
    


```python
!pip install matplotlib
```

    Requirement already satisfied: matplotlib in c:\users\user\anaconda3\lib\site-packages (3.1.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (2.4.6)
    Requirement already satisfied: numpy>=1.11 in c:\users\user\anaconda3\lib\site-packages (from matplotlib) (1.18.1)
    Requirement already satisfied: setuptools in c:\users\user\anaconda3\lib\site-packages (from kiwisolver>=1.0.1->matplotlib) (45.2.0.post20200210)
    Requirement already satisfied: six in c:\users\user\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib) (1.14.0)
    


```python
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
```

    Epoch 1/100
    100/100 [==============================] - 0s 994us/step - loss: 1.1712 - accuracy: 0.5263
    Epoch 2/100
    100/100 [==============================] - 0s 957us/step - loss: 1.1550 - accuracy: 0.5338
    Epoch 3/100
    100/100 [==============================] - 0s 957us/step - loss: 1.1470 - accuracy: 0.5338
    Epoch 4/100
    100/100 [==============================] - 0s 947us/step - loss: 1.1444 - accuracy: 0.5338
    Epoch 5/100
    100/100 [==============================] - 0s 957us/step - loss: 1.1358 - accuracy: 0.5338
    Epoch 6/100
    100/100 [==============================] - 0s 977us/step - loss: 1.1284 - accuracy: 0.5338
    Epoch 7/100
    100/100 [==============================] - 0s 947us/step - loss: 1.1212 - accuracy: 0.5338
    Epoch 8/100
    100/100 [==============================] - 0s 977us/step - loss: 1.1019 - accuracy: 0.5343
    Epoch 9/100
    100/100 [==============================] - 0s 948us/step - loss: 1.0882 - accuracy: 0.5338
    Epoch 10/100
    100/100 [==============================] - 0s 928us/step - loss: 1.0647 - accuracy: 0.5348
    Epoch 11/100
    100/100 [==============================] - 0s 947us/step - loss: 1.0471 - accuracy: 0.5438
    Epoch 12/100
    100/100 [==============================] - 0s 937us/step - loss: 1.0253 - accuracy: 0.5473
    Epoch 13/100
    100/100 [==============================] - 0s 938us/step - loss: 1.0035 - accuracy: 0.5563
    Epoch 14/100
    100/100 [==============================] - 0s 968us/step - loss: 0.9793 - accuracy: 0.5673
    Epoch 15/100
    100/100 [==============================] - 0s 918us/step - loss: 0.9590 - accuracy: 0.5703
    Epoch 16/100
    100/100 [==============================] - 0s 928us/step - loss: 0.9410 - accuracy: 0.5738
    Epoch 17/100
    100/100 [==============================] - 0s 957us/step - loss: 0.9101 - accuracy: 0.5913
    Epoch 18/100
    100/100 [==============================] - 0s 957us/step - loss: 0.8888 - accuracy: 0.6093
    Epoch 19/100
    100/100 [==============================] - 0s 947us/step - loss: 0.8592 - accuracy: 0.6138
    Epoch 20/100
    100/100 [==============================] - 0s 937us/step - loss: 0.8323 - accuracy: 0.6428
    Epoch 21/100
    100/100 [==============================] - 0s 937us/step - loss: 0.8168 - accuracy: 0.6608
    Epoch 22/100
    100/100 [==============================] - 0s 928us/step - loss: 0.7900 - accuracy: 0.6903
    Epoch 23/100
    100/100 [==============================] - 0s 947us/step - loss: 0.7734 - accuracy: 0.7024
    Epoch 24/100
    100/100 [==============================] - 0s 967us/step - loss: 0.7370 - accuracy: 0.7224
    Epoch 25/100
    100/100 [==============================] - 0s 928us/step - loss: 0.7220 - accuracy: 0.7244
    Epoch 26/100
    100/100 [==============================] - 0s 928us/step - loss: 0.6912 - accuracy: 0.7524
    Epoch 27/100
    100/100 [==============================] - 0s 977us/step - loss: 0.6879 - accuracy: 0.7414
    Epoch 28/100
    100/100 [==============================] - 0s 948us/step - loss: 0.6732 - accuracy: 0.7589
    Epoch 29/100
    100/100 [==============================] - 0s 947us/step - loss: 0.6524 - accuracy: 0.7689
    Epoch 30/100
    100/100 [==============================] - 0s 947us/step - loss: 0.6204 - accuracy: 0.7819
    Epoch 31/100
    100/100 [==============================] - 0s 928us/step - loss: 0.6157 - accuracy: 0.7854
    Epoch 32/100
    100/100 [==============================] - 0s 947us/step - loss: 0.6033 - accuracy: 0.7899
    Epoch 33/100
    100/100 [==============================] - 0s 947us/step - loss: 0.5795 - accuracy: 0.7954
    Epoch 34/100
    100/100 [==============================] - 0s 947us/step - loss: 0.5353 - accuracy: 0.8329
    Epoch 35/100
    100/100 [==============================] - 0s 947us/step - loss: 0.5432 - accuracy: 0.8379
    Epoch 36/100
    100/100 [==============================] - 0s 908us/step - loss: 0.5287 - accuracy: 0.8334
    Epoch 37/100
    100/100 [==============================] - 0s 957us/step - loss: 0.5331 - accuracy: 0.8174
    Epoch 38/100
    100/100 [==============================] - 0s 957us/step - loss: 0.5227 - accuracy: 0.8269
    Epoch 39/100
    100/100 [==============================] - 0s 947us/step - loss: 0.5029 - accuracy: 0.8399
    Epoch 40/100
    100/100 [==============================] - 0s 928us/step - loss: 0.4897 - accuracy: 0.8379
    Epoch 41/100
    100/100 [==============================] - 0s 918us/step - loss: 0.4780 - accuracy: 0.8489
    Epoch 42/100
    100/100 [==============================] - 0s 938us/step - loss: 0.4812 - accuracy: 0.8454
    Epoch 43/100
    100/100 [==============================] - 0s 908us/step - loss: 0.4873 - accuracy: 0.8449
    Epoch 44/100
    100/100 [==============================] - 0s 948us/step - loss: 0.4661 - accuracy: 0.8419
    Epoch 45/100
    100/100 [==============================] - 0s 938us/step - loss: 0.4551 - accuracy: 0.8654
    Epoch 46/100
    100/100 [==============================] - 0s 928us/step - loss: 0.4430 - accuracy: 0.8589
    Epoch 47/100
    100/100 [==============================] - 0s 928us/step - loss: 0.4350 - accuracy: 0.8574
    Epoch 48/100
    100/100 [==============================] - 0s 947us/step - loss: 0.4347 - accuracy: 0.8619
    Epoch 49/100
    100/100 [==============================] - 0s 928us/step - loss: 0.4317 - accuracy: 0.8544
    Epoch 50/100
    100/100 [==============================] - 0s 898us/step - loss: 0.4287 - accuracy: 0.8669
    Epoch 51/100
    100/100 [==============================] - 0s 927us/step - loss: 0.4037 - accuracy: 0.8769
    Epoch 52/100
    100/100 [==============================] - 0s 928us/step - loss: 0.4150 - accuracy: 0.8669
    Epoch 53/100
    100/100 [==============================] - 0s 957us/step - loss: 0.4099 - accuracy: 0.8644
    Epoch 54/100
    100/100 [==============================] - 0s 878us/step - loss: 0.4030 - accuracy: 0.8709
    Epoch 55/100
    100/100 [==============================] - 0s 918us/step - loss: 0.3936 - accuracy: 0.8769
    Epoch 56/100
    100/100 [==============================] - 0s 928us/step - loss: 0.3773 - accuracy: 0.8819
    Epoch 57/100
    100/100 [==============================] - 0s 937us/step - loss: 0.3705 - accuracy: 0.8859
    Epoch 58/100
    100/100 [==============================] - 0s 928us/step - loss: 0.3826 - accuracy: 0.8814
    Epoch 59/100
    100/100 [==============================] - 0s 957us/step - loss: 0.3849 - accuracy: 0.8699
    Epoch 60/100
    100/100 [==============================] - 0s 948us/step - loss: 0.3566 - accuracy: 0.8879
    Epoch 61/100
    100/100 [==============================] - 0s 908us/step - loss: 0.3685 - accuracy: 0.8754
    Epoch 62/100
    100/100 [==============================] - 0s 908us/step - loss: 0.3587 - accuracy: 0.8819
    Epoch 63/100
    100/100 [==============================] - 0s 928us/step - loss: 0.3766 - accuracy: 0.8794
    Epoch 64/100
    100/100 [==============================] - 0s 908us/step - loss: 0.3542 - accuracy: 0.8829
    Epoch 65/100
    100/100 [==============================] - 0s 928us/step - loss: 0.3555 - accuracy: 0.8934
    Epoch 66/100
    100/100 [==============================] - 0s 937us/step - loss: 0.3519 - accuracy: 0.8844
    Epoch 67/100
    100/100 [==============================] - 0s 937us/step - loss: 0.3412 - accuracy: 0.8979
    Epoch 68/100
    100/100 [==============================] - 0s 938us/step - loss: 0.3283 - accuracy: 0.8929
    Epoch 69/100
    100/100 [==============================] - 0s 957us/step - loss: 0.3373 - accuracy: 0.8899
    Epoch 70/100
    100/100 [==============================] - 0s 938us/step - loss: 0.3128 - accuracy: 0.9030
    Epoch 71/100
    100/100 [==============================] - 0s 947us/step - loss: 0.3132 - accuracy: 0.9105
    Epoch 72/100
    100/100 [==============================] - 0s 927us/step - loss: 0.3127 - accuracy: 0.9045
    Epoch 73/100
    100/100 [==============================] - 0s 947us/step - loss: 0.3196 - accuracy: 0.8989
    Epoch 74/100
    100/100 [==============================] - 0s 937us/step - loss: 0.3339 - accuracy: 0.8934
    Epoch 75/100
    100/100 [==============================] - 0s 938us/step - loss: 0.3017 - accuracy: 0.9050
    Epoch 76/100
    100/100 [==============================] - 0s 937us/step - loss: 0.3273 - accuracy: 0.8934
    Epoch 77/100
    100/100 [==============================] - 0s 918us/step - loss: 0.3119 - accuracy: 0.8944
    Epoch 78/100
    100/100 [==============================] - 0s 1ms/step - loss: 0.3103 - accuracy: 0.9020
    Epoch 79/100
    100/100 [==============================] - 0s 1ms/step - loss: 0.2931 - accuracy: 0.9080
    Epoch 80/100
    100/100 [==============================] - 0s 957us/step - loss: 0.2869 - accuracy: 0.9085
    Epoch 81/100
    100/100 [==============================] - 0s 957us/step - loss: 0.2715 - accuracy: 0.9155
    Epoch 82/100
    100/100 [==============================] - 0s 1ms/step - loss: 0.2928 - accuracy: 0.9040
    Epoch 83/100
    100/100 [==============================] - 0s 918us/step - loss: 0.2766 - accuracy: 0.9110
    Epoch 84/100
    100/100 [==============================] - 0s 908us/step - loss: 0.2834 - accuracy: 0.9120
    Epoch 85/100
    100/100 [==============================] - 0s 858us/step - loss: 0.2832 - accuracy: 0.9105
    Epoch 86/100
    100/100 [==============================] - 0s 898us/step - loss: 0.2917 - accuracy: 0.9075
    Epoch 87/100
    100/100 [==============================] - 0s 918us/step - loss: 0.2859 - accuracy: 0.9100
    Epoch 88/100
    100/100 [==============================] - 0s 858us/step - loss: 0.2980 - accuracy: 0.9080
    Epoch 89/100
    100/100 [==============================] - 0s 888us/step - loss: 0.2861 - accuracy: 0.9060
    Epoch 90/100
    100/100 [==============================] - 0s 888us/step - loss: 0.2859 - accuracy: 0.9130
    Epoch 91/100
    100/100 [==============================] - 0s 898us/step - loss: 0.2902 - accuracy: 0.9020
    Epoch 92/100
    100/100 [==============================] - 0s 888us/step - loss: 0.2705 - accuracy: 0.9175
    Epoch 93/100
    100/100 [==============================] - 0s 918us/step - loss: 0.2811 - accuracy: 0.9125
    Epoch 94/100
    100/100 [==============================] - 0s 908us/step - loss: 0.2611 - accuracy: 0.9175
    Epoch 95/100
    100/100 [==============================] - 0s 888us/step - loss: 0.2703 - accuracy: 0.9100
    Epoch 96/100
    100/100 [==============================] - 0s 858us/step - loss: 0.2472 - accuracy: 0.9160
    Epoch 97/100
    100/100 [==============================] - 0s 868us/step - loss: 0.2506 - accuracy: 0.9200
    Epoch 98/100
    100/100 [==============================] - 0s 898us/step - loss: 0.2743 - accuracy: 0.9065
    Epoch 99/100
    100/100 [==============================] - 0s 908us/step - loss: 0.2601 - accuracy: 0.9160
    Epoch 100/100
    100/100 [==============================] - 0s 908us/step - loss: 0.2648 - accuracy: 0.9065
    1/1 [==============================] - 0s 996us/step - loss: 0.3837 - accuracy: 0.9362
    test Acc: 0.936170220375061
    fizz
    other
    other
    fizz
    buzz
    


```python

```
