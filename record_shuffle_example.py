import json
import os

from tqdm import tqdm
import numpy as np
from datetime import datetime
import data_serialize
from fastdatasets.record import load_dataset as Loader,gfile,RECORD,DataType,WriterObject,FeatureWriter
import copy

class TimeSpan:
    def __init__(self,string=''):
        self.start(string)
    def start(self,string):
        self.string = string
        self.s = datetime.now()
        print(self.string,'..........')

    def show(self):
        e = datetime.now()
        print(self.string,': ',(e - self.s),'second: ', (e - self.s).seconds,'\n')


def shuffle_records(record_filenames,out_dir,out_record_num,compression_type='GZIP'):
    print('shuffle_records record...')
    time = TimeSpan()
    time.start('load RandomDataset')
    options = RECORD.TFRecordOptions(compression_type=compression_type)
    dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    data_size = len(dataset_reader)
    time.show()

    all_example = []
    for i in tqdm(range(data_size),desc='load records'):
        serialized = dataset_reader[i]
        all_example.append(serialized)
    dataset_reader.close()

    shuffle_idx = list(range(data_size))
    writers = [WriterObject(os.path.join(out_dir, 'record_gzip_shuffle_{}.record'.format(i)), options=options) for i in range(out_record_num)]
    for i in tqdm(shuffle_idx,desc='shuffle record'):
        example = all_example[i]
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


if __name__ == '__main__':
    src_dir='/tmp/raw_record'
    dst_dir = '/tmp/raw_record_shuffle'
    if not os.path.exists(dst_dir):
        gfile.makedirs(dst_dir)

    example_files = gfile.glob(os.path.join(src_dir, 'record*record'))
    shuffle_records(record_filenames=example_files, out_dir=dst_dir, out_record_num=2)

    # read and parse
    read_parse_records(gfile.glob(os.path.join(dst_dir, 'record*record')))