# fastdatasets tfrecords examples

##   install
pip install -U fastdatasets>=0.7.8



## 存储

目前支持加载的权重：
- <strong>tfrecord</strong> 支持压缩 , numpywriter 序列化反序列化
- <strong>leveldb</strong> 支持压缩 , numpywriter 序列化反序列化
- <strong>lmdb</strong> 不支持压缩 , numpywriter 序列化反序列化
- <strong>memory</strong> 内存读写器, numpywriter 序列化反序列化
- <strong>memory_raw</strong> 内存读写器 , 原始数据迭代器

## 1. 介绍

目前支持加载的数据形式：
- <strong>auto_writer</strong>  封装 tfrecord leveldb lmdb memory , 普通读写
- <strong>auto_parallel_writer</strong>  封装 tfrecord leveldb lmdb memory , 适配并行读写
- <strong>memory_readwriter_example</strong> numpy writer for memory 内存读写
- <strong>memory_raw_readwriter_example</strong> numpy writer for memory 原始数据内存读写
- <strong>record_numpywriter_example</strong>  numpy writer for tfrecord
- <strong>leveldb_readwriter_example</strong>  numpy writer for leveldb
- <strong>lmdb_readwriter_example</strong> numpy writer for lmdb
- <strong>record_writer_example</strong>  writer for tfrecord 兼容 tf
- <strong>record_shuffle_example</strong>  shuffle for tfrecord



## 2. numpy writer and reader for record

```python
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
```




## 3. numpy writer and reader for leveldb
```python
# @Time    : 2022/10/27 20:37
# @Author  : tk
import numpy as np
from tqdm import tqdm
from fastdatasets.leveldb import DB,load_dataset,WriterObject,DataType,StringWriter,JsonWriter,FeatureWriter,NumpyWriter

db_path = 'd:\\example_leveldb_numpy'

def test_write(db_path):
    options = DB.LeveldbOptions(create_if_missing=True,error_if_exists=False)
    f = NumpyWriter(db_path, options = options)
    keys,values = [],[]
    n = 30
    for i in range(n):
        train_node = {
            "index":np.asarray(i,dtype=np.int64),
            'image': np.random.rand(3,4),
            'labels': np.random.randint(0,21128,size=(10),dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }
        keys.append('input{}'.format(i))
        values.append(train_node)
        if (i+1) % 10000 == 0:
            f.put_batch(keys,values)
            keys.clear()
            values.clear()
    if len(keys):
        f.put_batch(keys, values)

    f.file_writer.put('total_num',str(n))
    f.close()



def test_random(db_path):
    options = DB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = load_dataset.RandomDataset(db_path,
                                        data_key_prefix_list=('input',),
                                        num_key='total_num',
                                        options = options)

    dataset = dataset.parse_from_numpy_writer().shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        d = dataset[i]
        print(i,d)

test_write(db_path)
test_random(db_path)

```


## 4. numpy writer and reader for lmdb
```python
# @Time    : 2022/10/27 20:37
# @Author  : tk

import numpy as np
from tqdm import tqdm
from fastdatasets.lmdb import DB,load_dataset,WriterObject,DataType,StringWriter,JsonWriter,FeatureWriter,NumpyWriter

db_path = 'd:\\example_lmdb_numpy'

def test_write(db_path):
    options = DB.LmdbOptions(env_open_flag = 0,
                env_open_mode = 0o664, # 8进制表示
                txn_flag = 0,
                dbi_flag = 0,
                put_flag = 0)

    f = NumpyWriter(db_path, options = options,map_size=1024 * 1024 * 1024)

    keys, values = [], []
    n = 30
    for i in range(n):
        train_node = {
            'image': np.random.rand(3, 4),
            'labels': np.random.randint(0, 21128, size=(10), dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }
        keys.append('input{}'.format(i))
        values.append(train_node)
        if (i + 1) % 10000 == 0:
            f.put_batch(keys, values)
            keys.clear()
            values.clear()
    if len(keys):
        f.put_batch(keys, values)

    f.file_writer.put('total_num', str(n))
    f.close()



def test_random(db_path):
    options = DB.LmdbOptions(env_open_flag=DB.LmdbFlag.MDB_RDONLY,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)
    dataset = load_dataset.RandomDataset(db_path,
                                        data_key_prefix_list=('input',),
                                        num_key='total_num',
                                        options = options)

    dataset = dataset.parse_from_numpy_writer().shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d = dataset[i]
        print(d)

test_write(db_path)
test_random(db_path)

```

## 5. numpy writer and reader for memory
```python
# @Time    : 2022/10/27 20:37
# @Author  : tk
import numpy as np
from tqdm import tqdm
from fastdatasets.memory import MEMORY,load_dataset,WriterObject,DataType,StringWriter,FeatureWriter,NumpyWriter

db_path = 'd:\\example_leveldb_numpy'

def test_write(db_path):
    options = MEMORY.MemoryOptions()
    f = NumpyWriter(db_path, options = options)
    values = []
    n = 30
    for i in range(n):
        train_node = {
            "index":np.asarray(i,dtype=np.int64),
            'image': np.random.rand(3,4),
            'labels': np.random.randint(0,21128,size=(10),dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }
        values.append(train_node)
        if (i+1) % 10000 == 0:
            f.write_batch(values)
            values.clear()
    if len(values):
        f.write_batch(values)
    real_data = f.file_writer.data()

    f.close()
    return real_data


def test_random(db_path):
    options = MEMORY.MemoryOptions()
    dataset = load_dataset.RandomDataset(db_path,options = options)

    dataset = dataset.parse_from_numpy_writer().shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        d = dataset[i]
        print(i,d)

db_path = test_write(db_path)
test_random(db_path)
```

## 5. numpy writer and reader for raw memory
```python
# @Time    : 2022/10/27 20:37
# @Author  : tk
import numpy as np
from tqdm import tqdm
from fastdatasets.memory import MEMORY,load_dataset,WriterObject,DataType,StringWriter,FeatureWriter,NumpyWriter

db_path = 'd:\\example_leveldb_numpy'

def test_write(db_path):
    options = MEMORY.MemoryOptions()
    f = WriterObject(db_path, options = options)
    values = []
    n = 30
    for i in range(n):
        train_node = {
            "index":np.asarray(i,dtype=np.int64),
            'image': np.random.rand(3,4),
            'labels': np.random.randint(0,21128,size=(10),dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }
        values.append(train_node)
        if (i+1) % 10000 == 0:
            f.write_batch(values)
            values.clear()
    if len(values):
        f.write_batch(values)
    real_data = f.file_writer.data()
    f.close()
    return real_data


def test_random(db_path):
    options = MEMORY.MemoryOptions()
    dataset = load_dataset.RandomDataset(db_path,options = options)

    dataset = dataset.shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        d = dataset[i]
        print(i,d)

db_path = test_write(db_path)
test_random(db_path)
```