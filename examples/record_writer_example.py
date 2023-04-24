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
#兼容 tensorflow.data
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