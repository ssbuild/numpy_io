# fastdatasets tfrecords examples

##  1. install
pip install -U fastdatasets

## 2. write records and shuffle records

```python
import json
import os
import random

import data_serialize
from tqdm import tqdm
import numpy as np
from datetime import datetime
from fastdatasets.record import load_dataset as Loader,gfile,RECORD,DataType,FeatureWriter

import copy


def write_records(data,out_dir,out_record_num,compression_type='GZIP'):
    print('write_records record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    # writers = [TFRecordWriter(os.path.join(out_dir, 'record_{}.gzip'.format(i)), options) for i in range(out_file_num)]
    writers = [FeatureWriter(os.path.join(out_dir, 'record_gzip_{}.record'.format(i)), options) for i in range(out_record_num)]
    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)

    for i in tqdm(shuffle_idx,desc='write record'):
        example = data[i]
        writers[i % out_record_num].write(example)
    for writer in writers:
        writer.close()


def read_parse_records(record_filenames,compression_type='GZIP'):
    print('read and parse record...')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.IterableDataset(record_filenames, options=options, with_share_memory=True)

    def parse_fn(x):
        example = data_serialize.Example()
        example.ParseFromString(x)
        feature = example.features.feature
        return feature

    dataset_reader = dataset_reader.apply(parse_fn)
    for example in dataset_reader:
        input_ids = example['input_ids'].int64_list
        seg_ids= example['seg_ids'].int64_list
        other = example['other'].bytes_list
        labels = example['labels'].bytes_list
        print(input_ids.value)
        print(seg_ids.value)
        print(other.value[0],other.value[1])
        print(labels.value[0])
        break

def get_data():
    labels = [0, 0, 0, 1]
    one_node = {
        'input_ids': {
            'dtype': DataType.int64_list,
            'data': np.random.randint(0, 21128, size=(512,), dtype=np.int32).tolist()
        },
        'seg_ids': {
            'dtype': DataType.int64_list,
            'data': np.zeros(shape=(512,), dtype=np.int32).tolist()
        },
        'other': {
            'dtype': DataType.bytes_list,
            'data': [b'aaaa', b'bbbbbbbbbbbb']
        },
        'labels': {
            'dtype': DataType.bytes_list,
            'data': [bytes(json.dumps(labels, ensure_ascii=True), encoding='utf-8')]
        }
    }

    num_gen = 50000
    print('gen {} data ....'.format(num_gen))
    data = [copy.deepcopy(one_node) for i in range(num_gen)]
    return data

if __name__ == '__main__':
    data = get_data()
    assert isinstance(data,list)
    #records
    record_dir = '/tmp/raw_record'
    if not os.path.exists(record_dir):
        gfile.makedirs(record_dir)
    write_records(data,out_dir=record_dir,out_record_num=3)

    #read and parse
    read_parse_records(gfile.glob(os.path.join(record_dir,'record*record')))
```



## 3. read records

```python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 9:36

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
```

## 4. leveldb dataset
```python
import random
from tqdm import tqdm
import numpy as np
import json
import copy
from fastdatasets.leveldb import DB,load_dataset as Loader, DataType,FeatureWriter,WriterObject,BytesWriter

db_path = 'd:\\example_leveldb'

def get_data():
    labels = np.asarray([0, 0, 0, 1],dtype=np.int32)
    one_node = {
        'image':  np.random.randint(0,256,size=(128,128),dtype=np.int32).tobytes(),
        'label': labels.tobytes()
    }
    num_gen = 100
    print('gen {} data ....'.format(num_gen))
    data = [copy.deepcopy(one_node) for i in range(num_gen)]
    return data

def write_data(db_path,data):
    options = DB.LeveldbOptions(create_if_missing=True, error_if_exists=False)
    writer = BytesWriter(db_path, options=options)
    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)
    n = len(shuffle_idx)

    keys,values=[],[]
    for i in tqdm(shuffle_idx, desc='write record'):
        example = data[i]
        for key,value in example.items():
            keys.append('{}{}'.format(key,i))
            values.append(value)

        if (i + 1) % 100000 == 0:
            writer.file_writer.put_batch(keys,values)
            keys.clear()
            values.clear()

    if len(keys):
        writer.file_writer.put_batch(keys, values)

    writer.file_writer.put('total_num', str(n))
    writer.close()

def test_read_random(db_path):
    options = DB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = Loader.RandomDataset(db_path,
                                         data_key_prefix_list=('image','label'),
                                         num_key='total_num',
                                         options=options)

    dataset = dataset.shuffle(-1)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d : dict = dataset[i]
        image,label = d.values()
        image = np.frombuffer(image,dtype=np.int32)
        image = image.reshape((128,128))

        label= np.frombuffer(label,dtype=np.int32)
        label = label.reshape((4,))
        print(image,label)
        break


if __name__ == '__main__':
    data = get_data()
    write_data(db_path,data)
    test_read_random(db_path)

```


## 4. lmdb dataset
```python

import random
from tqdm import tqdm
import numpy as np
import json
import copy
from fastdatasets.lmdb import DB,load_dataset as Loader, DataType,FeatureWriter,WriterObject,BytesWriter

db_path = 'd:\\example_lmdb'

def get_data():
    labels = np.asarray([0, 0, 0, 1],dtype=np.int32)
    one_node = {
        'image':  np.random.randint(0,256,size=(128,128),dtype=np.int32).tobytes(),
        'label': labels.tobytes()
    }
    num_gen = 100
    print('gen {} data ....'.format(num_gen))
    data = [copy.deepcopy(one_node) for i in range(num_gen)]
    return data

def write_data(db_path,data,map_size=1024 * 1024 * 1024):

    options = DB.LmdbOptions(env_open_flag=0,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)

    writer = BytesWriter(db_path, options=options,map_size=map_size)

    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)

    n = len(shuffle_idx)
    keys, values = [], []
    for i in tqdm(shuffle_idx, desc='write record'):
        example = data[i]
        for key, value in example.items():
            keys.append('{}{}'.format(key, i))
            values.append(value)

        if (i + 1) % 100000 == 0:
            writer.file_writer.put_batch(keys, values)
            keys.clear()
            values.clear()


    if len(keys):
        writer.file_writer.put_batch(keys, values)

    writer.file_writer.put('total_num', str(n))
    writer.close()


def test_read_random(db_path):
    options = DB.LmdbOptions(env_open_flag=DB.LmdbFlag.MDB_RDONLY,
                             env_open_mode=0o664,  # 8进制表示
                             txn_flag=0,
                             dbi_flag=0,
                             put_flag=0)
    dataset = Loader.RandomDataset(db_path,
                                         data_key_prefix_list=('image','label'),
                                         num_key='total_num',
                                         options=options)

    dataset = dataset.shuffle(-1)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d : dict = dataset[i]
        image,label = d.values()
        image = np.frombuffer(image,dtype=np.int32)
        image = image.reshape((128,128))

        label= np.frombuffer(label,dtype=np.int32)
        label = label.reshape((4,))
        print(image,label)
        break


if __name__ == '__main__':
    data = get_data()
    write_data(db_path,data)
    test_read_random(db_path)

```