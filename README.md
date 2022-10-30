# fastdatasets tfrecords examples

##  install
pip install -U fastdatasets


### 1.目录
    kv_reader_example.py    键值数据 dataset 读
    kv_writer_example.py    键值数据 dataset 写
    
    record_reader_example.py     record dataset 读
    record_writer_example.py     record dataset 写

## 2. write records and shuffle records

```python
import json
import os
import random

import data_serialize
from tqdm import tqdm
import numpy as np
from datetime import datetime
from fastdatasets.record_dataset import load_dataset,gfile,RECORD
from fastdatasets.writer.record import *

import copy

class TimeSpan:
    def start(self,string):
        self.string = string
        self.s = datetime.now()
        print(self.string,'..........')

    def show(self):
        e = datetime.now()
        print(self.string,': ',(e - self.s),'second: ', (e - self.s).seconds,'\n')

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

def shuffle_records(record_filenames,out_dir,out_record_num,compression_type='GZIP'):
    print('shuffle_records record...')
    time = TimeSpan()
    time.start('load RandomDataset')
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = load_dataset.RandomDataset(record_filenames, options=options, with_share_memory=True)
    data_size = len(dataset_reader)
    time.show()

    all_example = []
    for i in tqdm(range(data_size),desc='load records'):
        serialized = dataset_reader[i]
        all_example.append(serialized)
    dataset_reader.close()

    shuffle_idx = list(range(data_size))
    writers = [TFRecordWriter(os.path.join(out_dir, 'record_gzip_shuffle_{}.record'.format(i)), options=options) for i in range(out_record_num)]
    for i in tqdm(shuffle_idx,desc='shuffle record'):
        example = all_example[i]
        writers[i % out_record_num].write(example)
    for writer in writers:
        writer.close()

def read_parse_records(record_filenames,compression_type='GZIP'):
    print('read and parse record...')
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = load_dataset.IterableDataset(record_filenames, options=options, with_share_memory=True)

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

    record_num = 50000
    print('gen {} data ....'.format(record_num))
    data = [copy.deepcopy(one_node) for i in range(record_num)]
    return data

if __name__ == '__main__':
    data = get_data()
    assert isinstance(data,list)
    #records
    record_dir = '/tmp/raw_record'
    if not os.path.exists(record_dir):
        gfile.makedirs(record_dir)
    write_records(data,out_dir=record_dir,out_record_num=3)

    #shuffle records
    example_files = gfile.glob(os.path.join(record_dir,'record*record'))
    shuffle_record_dir = '/tmp/raw_record_shuffle'
    if not os.path.exists(shuffle_record_dir):
        gfile.makedirs(shuffle_record_dir)
    shuffle_records(record_filenames=example_files,out_dir=shuffle_record_dir,out_record_num=2)

    #read and parse
    read_parse_records(gfile.glob(os.path.join(shuffle_record_dir,'record*record')))
```



## 3. read records

```python
from fastdatasets.record_dataset import load_dataset,gfile,RECORD
from fastdatasets.writer.record import *
def read_iterable(record_filenames,compression_type='GZIP'):
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = load_dataset.IterableDataset(record_filenames, options=options, with_share_memory=True)

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
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = load_dataset.RandomDataset(record_filenames, options=options, with_share_memory=True)
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

from tqdm import tqdm
from fastdatasets.writer.leveldb import LEVELDB_writer
from fastdatasets.leveldb_dataset import LEVELDB,load_dataset

db_path = 'd:\\example_leveldb'


def test_write(db_path):
    options = LEVELDB.LeveldbOptions(create_if_missing=True,error_if_exists=False)
    f = LEVELDB_writer(db_path, options = options)
    
    n = 0
    for i in range(30):
        f.put('input{}'.format(i).encode(encoding='utf-8'), str(i))
        f.put('label{}'.format(i).encode(), str(i))
        n += 1
    f.put('total_num',str(n))
    f.close()


def test_iterable(db_path):
    options = LEVELDB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = load_dataset.IterableDataset(db_path, options = options)
    for d in dataset:
        print(d)

def test_random(db_path):
    options = LEVELDB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = load_dataset.RandomDataset(db_path,
                                        data_key_prefix_list=('input','label'),
                                        num_key='total_num',
                                        options = options)

    dataset = dataset.shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        d = dataset[i]
        print(i,d)

test_write(db_path)
test_iterable(db_path)
test_random(db_path)

```


## 4. lmdb dataset
```python

from tqdm import tqdm
from fastdatasets.writer.lmdb import LMDB_writer
from fastdatasets.lmdb_dataset import LMDB,load_dataset

db_path = 'd:\\example_lmdb_new2'


def test_write(db_path):
    options = LMDB.LmdbOptions(env_open_flag = 0,
                env_open_mode = 0o664, # 8进制表示
                txn_flag = 0,
                dbi_flag = 0,
                put_flag = 0)

    f = LMDB_writer(db_path, options = options,map_size=1024 * 1024 * 1024)
    
    n = 0
    for i in range(20):
        f.put('input{}'.format(i).encode(encoding='utf-8'), str(i))
        f.put('label{}'.format(i).encode(), str(i))
        n += 1
    f.put('total_num',str(n))
    f.close()


def test_iterable(db_path):
    options = LMDB.LmdbOptions(env_open_flag=LMDB.LmdbFlag.MDB_RDONLY,
                     env_open_mode=0o664,  # 8进制表示
                     txn_flag=0,
                     dbi_flag=0,
                     put_flag=0)
    dataset = load_dataset.IterableDataset(db_path,options = options)
    for d in dataset:
        print(d)

def test_random(db_path):
    options = LMDB.LmdbOptions(env_open_flag=LMDB.LmdbFlag.MDB_RDONLY,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)
    dataset = load_dataset.RandomDataset(db_path,
                                        data_key_prefix_list=('input','label'),
                                        num_key='total_num',
                                        options = options)

    dataset = dataset.shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        d = dataset[i]
        print(i,d)

test_write(db_path)
test_iterable(db_path)
test_random(db_path)

```