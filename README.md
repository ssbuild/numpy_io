# fastdatasets tfrecords examples

## 1. 安装
pip install -U fastdatasets


## 2. 写大文件 shuffle 数据

```python
import json
import os
import random

import data_serialize
from tqdm import tqdm
import numpy as np
from datetime import datetime
from fastdatasets import TFRecordOptions,TFRecordWriter,RecordLoader,FeatrueWriter,DataType,gfile
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
    options = TFRecordOptions(compression_type=compression_type)
    # writers = [TFRecordWriter(os.path.join(out_dir, 'record_{}.gzip'.format(i)), options) for i in range(out_file_num)]
    writers = [FeatrueWriter(os.path.join(out_dir, 'record_{}.gzip'.format(i)), options) for i in range(out_record_num)]
    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)

    for i,id in enumerate(tqdm(shuffle_idx,desc='write record')):
        example = data[id]
        writers[i % out_record_num].write(example)
    for writer in writers:
        writer.close()

def shuffle_records(record_filenames,out_dir,out_record_num,compression_type='GZIP'):
    print('shuffle_records record...')
    time = TimeSpan()
    time.start('load RandomDataset')
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = RecordLoader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    data_size = len(dataset_reader)
    time.show()

    all_example = []
    for i, id in enumerate(tqdm(range(data_size))):
        serialized = dataset_reader[id]
        all_example.append(serialized)
    dataset_reader.close()

    shuffle_idx = list(range(data_size))
    writers = [TFRecordWriter(os.path.join(out_dir, 'record_{}.gzip'.format(i)), options=options) for i in range(out_record_num)]
    for i, id in enumerate(tqdm(shuffle_idx,desc='shuffle record')):
        example = all_example[id]
        writers[i % out_record_num].write(example)
    for writer in writers:
        writer.close()

def read_data(record_filenames,compression_type='GZIP'):
    print('read and parse record...')
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = RecordLoader.IterableDataset(record_filenames, options=options, with_share_memory=True)

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

if __name__ == '__main__':
    labels = [0,0,0,1]
    node = {
        'input_ids': {
            'dtype': DataType.int64_list,
            'data': np.random.randint(0, 21128, size=(512,),dtype=np.int32).tolist()
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
            'data': [bytes(json.dumps(labels,ensure_ascii=True),encoding='utf-8')]
        }
    }

    record_num = 50000
    print('gen {} data ....'.format(record_num))
    data = [copy.deepcopy(node) for i in range(record_num)]

    out_dir = '/tmp/raw_record'
    if not os.path.exists(out_dir):
        gfile.makedirs(out_dir)
    write_records(data,out_dir=out_dir,out_record_num=4)
    #shuffle
    in_dir = '/tmp/raw_record/record*gzip'
    example_files = gfile.glob(in_dir)
    out_dir = '/tmp/raw_record_shuffle'
    if not os.path.exists(out_dir):
        gfile.makedirs(out_dir)
    shuffle_records(record_filenames=example_files,out_dir=out_dir,out_record_num=2)

    #读取
    in_dir = '/tmp/raw_record_shuffle/record*gzip'
    read_data( gfile.glob(in_dir))
```



## 3. 读取大文件多文件records

```python
from fastdatasets import TFRecordOptions,RecordLoader,FeatrueWriter,DataType,gfile

def read_data(record_filenames,compression_type='GZIP'):
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = RecordLoader.IterableDataset(record_filenames, options=options, with_share_memory=True)

    i = 0
    for example in dataset_reader:
        if i % 1000 == 0:
            print(example)
        i += 1

def read_data2(record_filenames,compression_type='GZIP'):
    options = TFRecordOptions(compression_type=compression_type)
    dataset_reader = RecordLoader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    example_size = len(dataset_reader)
    for i in range(example_size):
        example = dataset_reader[i]
        if i % 1000 == 0:
            print(example)
```