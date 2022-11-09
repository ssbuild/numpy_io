# fastdatasets tfrecords examples

##   install
pip install -U fastdatasets


```text
support backend as flow
tfrecord:  support compression
leveldb: support compression
lmdb: doe not support compression

easy method
    record_numpywriter_example.py
    lmdb_readwriter_example.py
    leveldb_readwriter_example.py

complex method and this method is compatible with tensorflow.datasets
    record_writer_example.py
    record_shuffle_example.py
    record_reader_example.py
```

## 1. unity writers
```python
# @Time    : 2022/11/6 10:40
import typing
import numpy as np
from enum import Enum
from fastdatasets.utils import parallel_apply,C_parallel_node
from fastdatasets.record import load_dataset as record_loader,writer as record_writer,RECORD
from fastdatasets.leveldb import load_dataset as leveldb_loader,writer as leveldb_writer,LEVELDB
from fastdatasets.lmdb import load_dataset as lmdb_loader,writer as lmdb_writer,LMDB
from transformers import BertTokenizer

class E_file_backend(Enum):
    record = 0
    leveldb = 1
    lmdb = 2

    @staticmethod
    def from_string(b: str):
        b = b.lower()
        if b == 'record':
            return E_file_backend.record
        elif b == 'leveldb':
            return E_file_backend.leveldb
        elif b == 'lmdb':
            return E_file_backend.lmdb
        return None


class Parallel_workers(C_parallel_node):
    def __init__(self,filename,backend,fn_input_hook: typing.Callable,user_data:typing.Any,*args,**kwargs):
        super(Parallel_workers, self).__init__(*args,**kwargs)
        self.backend = backend
        self.write_fn = None
        self.batch_count = 2000
        if self.backend == E_file_backend.record:
            self.batch_count = 2000
            self.f_writer = record_writer.NumpyWriter(filename, options=RECORD.TFRecordOptions(compression_type='GZIP'))

        elif self.backend == E_file_backend.leveldb:
            self.batch_count = 100000
            self.f_writer = leveldb_writer.NumpyWriter(filename, options=LEVELDB.LeveldbOptions(create_if_missing=True,
                                                                                                  error_if_exists=False,
                                                                                                  write_buffer_size=1024 * 1024 * 512))
        elif self.backend == E_file_backend.lmdb:
            self.batch_count = 100000
            self.f_writer = lmdb_writer.NumpyWriter(filename, options = LMDB.LmdbOptions(env_open_flag = 0,
                env_open_mode = 0o664, # 8进制表示
                txn_flag = 0,
                dbi_flag = 0,
                put_flag = 0),map_size=1024 * 1024 * 1024 * 150)


        self.batch_keys = []
        self.batch_values = []
        self.total_num = 0
        self.fn_input_hook = fn_input_hook
        self.user_data = user_data

    def write_batch_data(self):
        self.total_num += len(self.batch_values)
        if self.backend == E_file_backend.record:
            self.f_writer.write_batch(self.batch_values)
        elif self.backend == E_file_backend.leveldb or self.backend == E_file_backend.lmdb:
            self.f_writer.put_batch(self.batch_keys,self.batch_values)
        self.batch_keys.clear()
        self.batch_values.clear()

    #继承
    def on_input_process(self, index,x):
        return self.fn_input_hook(index,x,self.user_data)

    # 继承
    def on_output_process(self, index, x):
        self.batch_keys.append('input{}'.format(index))
        self.batch_values.append(x)
        if len(self.batch_values) % self.batch_count == 0:
            self.write_batch_data()

    # 继承
    def on_output_cleanup(self):
        if self.f_writer is not None:
            if len(self.batch_values) > 0:
                self.write_batch_data()
            if self.backend != E_file_backend.record:
                self.f_writer.file_writer.put('total_num', str(self.total_num))
            self.f_writer = None



class DataWriteHelper:
    def __init__(self,backend='record',num_writer_worker=8,max_seq_length=512):
        assert backend in ['record', 'lmdb', 'leveldb']
        self._backend_type = backend
        self._backend = E_file_backend.from_string(backend)
        self.num_writer_worker = num_writer_worker
        self.max_seq_length = max_seq_length
        assert self.num_writer_worker > 0

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        self._backend = value

    @property
    def backend_type(self):
        return self._backend_type

    @backend_type.setter
    def backend_type(self, value):
        self._backend_type = value

    # 多进程写大文件
    def save(self, cache_file: str, data: list, tokenizer: BertTokenizer):
        user_data = (tokenizer,)
        worker_node = Parallel_workers(cache_file,self.backend,self.tokenize_data,user_data,num_process_worker=self.num_writer_worker)
        parallel_apply(data, worker_node)

    #切分词
    def tokenize_data(self,data_index: int, data: typing.Any,user_data: typing.Any):
        tokenizer: BertTokenizer
        tokenizer = user_data[0] if isinstance(user_data,tuple) else user_data
        max_seq_length = self.max_seq_length
        x = data
        if isinstance(x, tuple):
            o = tokenizer(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True,
                          add_special_tokens=True)
        else:
            o = tokenizer(x, max_length=max_seq_length, truncation=True, add_special_tokens=True, )

        input_ids = o['input_ids']
        attention_mask = o['attention_mask']
        token_type_ids = o['token_type_ids']

        input_length = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = self.max_seq_length - input_length
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            token_type_ids = np.pad(token_type_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        node = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids':token_type_ids,
            'seqlen': input_length
        }

        return node
    #读取文件
    def read_from_file(self, filename):
        D = []
        with open(filename,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r\n','').replace('\n')
                D.append(line)
        return D



class DataReadLoader:
    @staticmethod
    def load(filename: typing.Union[typing.List[str], str],backend: str):
        backend = E_file_backend.from_string(backend)
        if backend == E_file_backend.record:
            dataset = record_loader.IterableDataset(filename,
                                                    options=RECORD.TFRecordOptions(compression_type='GZIP'))
        elif backend == E_file_backend.leveldb:
            dataset = leveldb_loader.RandomDataset(filename,
                                                   data_key_prefix_list=('input',),
                                                   num_key='total_num',
                                                   options=LEVELDB.LeveldbOptions(create_if_missing=True,
                                                                                  error_if_exists=False))
        elif backend == E_file_backend.lmdb:
            dataset = lmdb_loader.RandomDataset(filename,
                                                data_key_prefix_list=('input',),
                                                num_key='total_num',
                                                options=LMDB.LmdbOptions(env_open_flag=LMDB.LmdbFlag.MDB_RDONLY,
                                                                         env_open_mode=0o664,  # 8进制表示
                                                                         txn_flag=0,
                                                                         dbi_flag=0,
                                                                         put_flag=0),
                                                )
        else:
            dataset = None

        dataset = dataset.parse_from_numpy_writer().apply(DataReadLoader.dataset_hook)
        return dataset

    # 读取数据对齐 max_seq_length
    @staticmethod
    def dataset_hook(x: dict):
        d = {}
        for k in x:
            d[k] = np.asarray(x[k])
        return d

def make_dataset(tokenizer,outputfile,data_backend):
    dataHelper = DataWriteHelper(backend=data_backend, num_writer_worker=8, max_seq_length=64)
    # filename = './data.txt'
    # data = dataHelper.read_from_file(filename)
    data = [str(i) + 'fastdatasets numpywriter demo' for i in range(1000)]
    dataHelper.save(outputfile, data, tokenizer)

if __name__ == '__main__':
    data_backend = 'record'
    outputfile = './data.record'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    make_dataset(tokenizer,outputfile,data_backend)

    dataset = DataReadLoader.load(outputfile,data_backend)
    try:
        length = len(dataset)
    except:
        length = None

    if length is None:
        for d in dataset:
            print(d)
            break
    else:
        for i in range(length):
            print(dataset[i])

```

## 2. write records 

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




## 3. leveldb dataset
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


## 4. lmdb dataset
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