# -*- coding: utf-8 -*-
# @Time:  22:34
# @Author: tk
# @File：numpyadapter

import copy
import typing
import warnings
from enum import Enum
from fastdatasets.utils.py_features import Final
from fastdatasets.record import writer as record_writer, RECORD, load_dataset as record_loader
from fastdatasets.leveldb import writer as leveldb_writer, LEVELDB, load_dataset as leveldb_loader
from fastdatasets.lmdb import writer as lmdb_writer, LMDB, load_dataset as lmdb_loader
from fastdatasets.memory import writer as memory_writer, MEMORY, load_dataset as memory_loader
from .parallel import ParallelStruct, parallel_apply


__all__ = [
    'E_file_backend',
    'NumpyWriterAdapter',
    'NumpyReaderAdapter',
    'ParallelStruct',
    'parallel_apply',
    'ParallelNumpyWriter'
]


class E_file_backend(Enum):
    record = 0
    leveldb = 1
    lmdb = 2
    memory = 3
    memory_raw = 4

    @staticmethod
    def from_string(b: str):
        b = b.lower()
        if b == 'record':
            return E_file_backend.record
        elif b == 'leveldb':
            return E_file_backend.leveldb
        elif b == 'lmdb':
            return E_file_backend.lmdb
        elif b == 'memory':
            return E_file_backend.memory
        elif b == 'memory_raw':
            return E_file_backend.memory_raw
        return None

    def to_string(self, b):
        if b == E_file_backend.record:
            return 'record'
        if b == E_file_backend.leveldb:
            return 'leveldb'
        if b == E_file_backend.lmdb:
            return 'lmdb'
        if b == E_file_backend.memory:
            return 'memory'
        if b == E_file_backend.memory_raw:
            return 'memory_raw'
        return None


class NumpyWriterAdapter:
    def __init__(self, filename: typing.Union[str, typing.List],
                 backend: typing.Union[E_file_backend, str],
                 options: typing.Union[
                     RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions, MEMORY.MemoryOptions] = None,
                 leveldb_write_buffer_size=1024 * 1024 * 512,
                 leveldb_max_file_size=10 * 1024 * 1024 * 1024,
                 lmdb_map_size=1024 * 1024 * 1024 * 150):

        self.filename = filename
        if isinstance(backend, E_file_backend):
            self._backend_type = E_file_backend.to_string(backend)
            self._backend = backend
        else:
            self._backend_type = backend
            self._backend = E_file_backend.from_string(backend)
        self._batch_buffer_num = 2000
        self._kv_flag = True
        if self._backend == E_file_backend.record:
            self._kv_flag = False
            self._batch_buffer_num = 2000
            if options is None:
                options = RECORD.TFRecordOptions(compression_type='GZIP')
            self._f_writer = record_writer.NumpyWriter(filename, options=options)

        elif self._backend == E_file_backend.leveldb:
            self._batch_buffer_num = 100000
            if options is None:
                options = LEVELDB.LeveldbOptions(create_if_missing=True,
                                                 error_if_exists=False,
                                                 write_buffer_size=leveldb_write_buffer_size,
                                                 max_file_size=leveldb_max_file_size)
            self._f_writer = leveldb_writer.NumpyWriter(filename, options=options)
        elif self._backend == E_file_backend.lmdb:
            self._batch_buffer_num = 100000
            if options is None:
                options = LMDB.LmdbOptions(env_open_flag=0,
                                           env_open_mode=0o664,  # 8进制表示
                                           txn_flag=0,
                                           dbi_flag=0,
                                           put_flag=0)
            self._f_writer = lmdb_writer.NumpyWriter(filename, options=options,
                                                     map_size=lmdb_map_size)
        elif self._backend == E_file_backend.memory:
            self._kv_flag = False
            self._batch_buffer_num = 100000
            if options is None:
                options = MEMORY.MemoryOptions()
            self._f_writer = memory_writer.NumpyWriter(filename, options=options)
        elif self._backend == E_file_backend.memory_raw:
            self._kv_flag = False
            self._batch_buffer_num = 100000
            if options is None:
                options = MEMORY.MemoryOptions()
            self._f_writer = memory_writer.WriterObject(filename, options=options)
        else:
            raise ValueError(
                'NumpyWriterAdapter does not support backend={} , not in record,leveldb,lmdb,memory,meory_raw'.format(
                    backend))

    def __del__(self):
        self.close()

    def close(self):
        if self._f_writer is not None:
            self._f_writer.close()
            self._f_writer = None

    @property
    def writer(self):
        return self._f_writer

    @property
    def is_kv_writer(self):
        return self._kv_flag

    @property
    def backend(self):
        return self._backend

    @property
    def backend_type(self):
        return self._backend_type

    @property
    def advice_batch_buffer_size(self):
        return self._batch_buffer_num


class NumpyReaderAdapter:
    @staticmethod
    def load(input_files: typing.Union[typing.List[str], str, typing.List[typing.Any]],
             backend: typing.Union[E_file_backend, str],
             options: typing.Union[
                 RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions, MEMORY.MemoryOptions] = None,
             data_key_prefix_list=('input',),
             num_key='total_num',
             cycle_length=1,
             block_length=1,
             with_record_iterable_dataset=True,
             with_parse_from_numpy=True):
        '''
            input_files: 文件列表
            backend: 存储引擎类型
            options: 存储引擎选项
            data_key_prefix_list: 键值数据库 键值前缀
            num_key: 键值数据库，记录数据总数建
            with_record_iterable_dataset 打开iterable_dataset
            with_parse_from_numpy 解析numpy数据
        '''

        parse_flag = True
        data_backend = backend if isinstance(backend, E_file_backend) else E_file_backend.from_string(backend)
        if data_backend == E_file_backend.record:
            if options is None:
                options = RECORD.TFRecordOptions(compression_type='GZIP')
            if with_record_iterable_dataset:
                dataset = record_loader.IterableDataset(input_files, cycle_length=cycle_length,
                                                        block_length=block_length, options=options)
            else:
                dataset = record_loader.RandomDataset(input_files, options=options)

        elif data_backend == E_file_backend.leveldb:
            if options is None:
                options = LEVELDB.LeveldbOptions(create_if_missing=True, error_if_exists=False)
            dataset = leveldb_loader.RandomDataset(input_files,
                                                   data_key_prefix_list=data_key_prefix_list,
                                                   num_key=num_key,
                                                   options=options)
        elif data_backend == E_file_backend.lmdb:
            if options is None:
                options = LMDB.LmdbOptions(env_open_flag=LMDB.LmdbFlag.MDB_RDONLY,
                                           env_open_mode=0o664,  # 8进制表示
                                           txn_flag=0,
                                           dbi_flag=0,
                                           put_flag=0)
            dataset = lmdb_loader.RandomDataset(input_files,
                                                data_key_prefix_list=data_key_prefix_list,
                                                num_key=num_key,
                                                options=options)
        elif data_backend == E_file_backend.memory:
            if options is None:
                options = MEMORY.MemoryOptions()
            dataset = memory_loader.RandomDataset(input_files, options=options)
        elif data_backend == E_file_backend.memory_raw:
            parse_flag = False
            if options is None:
                options = MEMORY.MemoryOptions()
            dataset = memory_loader.RandomDataset(input_files, options=options)
        else:
            dataset = None
        if with_parse_from_numpy and parse_flag:
            dataset = dataset.parse_from_numpy_writer()
        return dataset


class ParallelNumpyWriter(ParallelStruct, metaclass=Final):
    def __init__(self, *args, **kwargs):
        ParallelStruct.__init__(self, *args, **kwargs)
        self.batch_keys = []
        self.batch_values = []
        self.total_num = 0
        self.numpy_writer = None

    def open(self, outfile: typing.Union[str, typing.List],
             backend: typing.Union[E_file_backend, str],
             options: typing.Union[
                 RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions, MEMORY.MemoryOptions] = None,
             leveldb_write_buffer_size=1024 * 1024 * 512,
             lmdb_map_size=1024 * 1024 * 1024 * 150):

        self.numpy_writer = NumpyWriterAdapter(outfile, backend, options, leveldb_write_buffer_size, lmdb_map_size)
        self.backend = self.numpy_writer.backend
        self.backend_type = self.numpy_writer.backend_type
        self.is_kv_writer = self.numpy_writer.is_kv_writer
        self.write_batch_size = self.numpy_writer.advice_batch_buffer_size

    def write(self, data, input_hook_fn: typing.Callable, fn_args: typing.Union[typing.Tuple, typing.Dict],
              write_batch_size=None):
        self.input_hook_fn = input_hook_fn
        self.fn_args = fn_args

        assert self.numpy_writer is not None
        assert self.input_hook_fn is not None

        if write_batch_size is None or write_batch_size <= 0:
            write_batch_size = self.numpy_writer.advice_batch_buffer_size
            if write_batch_size >= len(data):
                write_batch_size = len(data) // 2

        if write_batch_size <= 0:
            write_batch_size = 1

        self.write_batch_size = write_batch_size
        parallel_apply(data, self)

    def flush(self):
        if not self.is_kv_writer:
            if self.backend == E_file_backend.memory_raw:
                self.numpy_writer.writer.write_batch([d for d in self.batch_values])
            else:
                self.numpy_writer.writer.write_batch(self.batch_values)

        else:
            self.numpy_writer.writer.put_batch(self.batch_keys, self.batch_values)
        self.batch_keys.clear()
        self.batch_values.clear()

    # 继承
    def on_input_process(self, x):
        return self.input_hook_fn(x, self.fn_args)

    # 继承
    def on_output_process(self, x):
        if x is None:
            return
        # 返回多个结果
        if isinstance(x, (list, tuple)):
            for one in x:
                self.batch_keys.append('input{}'.format(self.total_num))
                self.batch_values.append(one)
                self.total_num += 1
        # 返回一个结果
        else:
            self.batch_keys.append('input{}'.format(self.total_num))
            self.batch_values.append(x)
            self.total_num += 1

        if len(self.batch_values) > 0 and len(self.batch_values) % self.write_batch_size == 0:
            self.flush()

    # 继承
    def on_output_cleanup(self):
        if self.numpy_writer is not None:
            result = None
            if len(self.batch_values) > 0:
                self.flush()
            if self.is_kv_writer:
                self.numpy_writer.writer.file_writer.put('total_num', str(self.total_num))
            self.numpy_writer.close()
            self.numpy_writer = None

