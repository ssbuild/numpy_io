# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 9:36
import random

from fastdatasets.record import load_dataset as Loader,gfile,RECORD

def read_iterable(record_filenames,compression_type='GZIP'):
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.IterableDataset(record_filenames, options=options, with_share_memory=True)

    i = 0
    for example in dataset_reader:
        if i % 1000 == 0:
            print(example)
        i += 1
    # dataset 变换操作
    dataset_reader = dataset_reader.apply(lambda x:x).repeat(2).shuffle(128)
    i = 0
    for example in dataset_reader:
        if i % 1000 == 0:
            print(example)
        i += 1

def read_random(record_filenames,compression_type='GZIP'):
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    example_size = len(dataset_reader)
    for i in range(example_size):
        example = dataset_reader[i]
        if i % 1000 == 0:
            print(example)
    # dataset 变换操作
    dataset_reader = dataset_reader.apply(lambda x: x).repeat(2).shuffle(128)
    i = 0
    for example in dataset_reader:
        if i % 1000 == 0:
            print(example)
        i += 1