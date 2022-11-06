# @Time    : 2022/11/6 10:40
# @Author  : tk

import io
import copy
import json
import numpy as np
import data_serialize
from fastdatasets.record import load_dataset as Loader,FeatureWriter,RECORD,DataType
from fastdatasets.utils import parallel_apply,C_parallel_node

def write_record(record_filename,data,num_worker=10):
    class C_worker_node(C_parallel_node):
        f_writer = FeatureWriter(record_filename, options=RECORD.TFRecordOptions(compression_type='GZIP'))
        batch = []

        def on_coming(self, x):
            input_ids,labels = x
            if isinstance(input_ids,np.ndarray):
                input_ids = input_ids.tolist()
            if isinstance(labels,np.ndarray):
                labels = labels.tolist()
            node = {
                'input_ids': {
                    'dtype': DataType.int64_list,
                    'data': input_ids
                },
                'labels': {
                    'dtype': DataType.bytes_list,
                    'data': [bytes(json.dumps(labels, ensure_ascii=True), encoding='utf-8')]
                }
            }
            return node

        def on_output(self, x):
            self.batch.append(x)
            if len(self.batch) % 2000 == 0:
                self.f_writer.write_batch(self.batch)
                self.batch.clear()

        def on_done(self):
            if self.f_writer is not None:
                if len(self.batch) > 0:
                    self.f_writer.write_batch(self.batch)
                    self.batch.clear()
                self.f_writer.close()
                self.f_writer = None

    worker_node = C_worker_node(num_process_worker=num_worker,shuffle=True)
    parallel_apply(data, worker_node)


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
        labels = example['labels'].bytes_list
        print(input_ids.value)
        print(labels.value[0])
        break

if __name__ == '__main__':
    labels = [0, 0, 0, 1]
    input_ids = np.random.randint(0, 21128, size=(512,), dtype=np.int32)
    one = (input_ids,labels)
    num_gen = 50000
    print('gen {} data ....'.format(num_gen))
    data = [copy.deepcopy(one) for i in range(num_gen)]

    record_filename = '/tmp/example.record'
    write_record(record_filename,data,num_worker=10)
    read_parse_records(record_filename)