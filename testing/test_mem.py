# -*- coding: utf-8 -*-
# @Time    : 2022/11/18 15:33
import random


from fastdatasets.memory import load_dataset


data = list(range(100))
dataset_reader = load_dataset.RandomDataset(data)
train,devs = dataset_reader.split(0.7)
dev,test = devs.split(0.5)

t = []
for i in range(len(train)):
    t.append(train[i])
print(len(t),t)

t = []
for i in range(len(dev)):
    t.append(dev[i])
print(len(t),t)

t = []
for i in range(len(test)):
    t.append(test[i])
print(len(t),t)


train = train.concat([test])
t = []
for i in range(len(train)):
    t.append(train[i])
print(len(t),t)