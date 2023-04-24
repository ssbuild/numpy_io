# @Time    : 2022/9/18 23:27
import pickle
import data_serialize
import numpy as np
from fastdatasets.record import load_dataset
from fastdatasets.record import RECORD, WriterObject,FeatureWriter,StringWriter,PickleWriter,DataType,NumpyWriter

filename= r'd:\\example_writer.record'

def test_writer(filename):
    print('test_feature ...')
    options = RECORD.TFRecordOptions(compression_type='GZIP')
    f = NumpyWriter(filename,options=options)

    values = []
    n = 30
    for i in range(n):
        train_node = {
            "index": np.asarray(i, dtype=np.int64),
            'image': np.random.rand(3, 4),
            'labels': np.random.randint(0, 21128, size=(10), dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }

        values.append(train_node)
        if (i + 1) % 10000 == 0:
            f.write_batch( values)
            values.clear()
    if len(values):
        f.write_batch(values)
    f.close()

def test_iterable(filename):
    options = RECORD.TFRecordOptions(compression_type='GZIP')
    datasets = load_dataset.IterableDataset(filename, options=options).parse_from_numpy_writer()
    for i, d in enumerate(datasets):
        print(i, d)

def test_random(filename):
    options = RECORD.TFRecordOptions(compression_type='GZIP')
    datasets = load_dataset.RandomDataset(filename, options=options).parse_from_numpy_writer()
    print(len(datasets))
    for i in range(len(datasets)):
        d = datasets[i]
        print(i, d)

test_writer(filename)
test_iterable(filename)