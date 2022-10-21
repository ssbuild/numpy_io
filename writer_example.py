import json
import os
import random

import data_serialize
from tqdm import tqdm
import numpy as np
from datetime import datetime
from fastdatasets import TFRecordOptions,TFRecordWriter,RecordLoader,FeatureWriter,DataType,gfile
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
    dataset_reader = RecordLoader.RandomDataset(record_filenames, options=options, with_share_memory=True)
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