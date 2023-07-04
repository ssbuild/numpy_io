# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 9:16


import typing

from .numpyadapter import NumpyWriterAdapter, ParallelNumpyWriter,E_file_backend

__all__ = [
    'DataWriteHelper',
    'NumpyWriterAdapter',
    'ParallelNumpyWriter'
]

class DataWriteHelper:
    def __init__(self,
                 input_fn: typing.Callable[[typing.Any, tuple], typing.Union[typing.Dict, typing.List, typing.Tuple]],
                 input_fn_args: typing.Union[typing.Tuple,typing.Dict],
                 outfile: typing.Union[str,list],
                 backend='record',
                 num_process_worker=0,
                 shuffle=True):
        assert E_file_backend.from_string(backend) is not None

        self.input_fn = input_fn
        self.input_fn_args = input_fn_args
        self.outfile = outfile
        self._backend_type = backend
        self._parallel_writer = ParallelNumpyWriter(num_process_worker=num_process_worker,shuffle=shuffle)

    @property
    def backend_type(self):
        return self._backend_type

    @backend_type.setter
    def backend_type(self, value):
        self._backend_type = value

    # 多进程写大文件
    def save(self,data: list,
             options=None,
             parquet_options: typing.Optional = None,
             schema: typing.Optional[typing.Dict] = None,
             leveldb_write_buffer_size=1024 * 1024 * 512,
             leveldb_max_file_size=10 * 1024 * 1024 * 1024,
             lmdb_map_size=1024 * 1024 * 1024 * 150,
             batch_size=None
             ):

        self._parallel_writer.open(self.outfile ,
                                   backend=self.backend_type,
                                   schema=schema,
                                   parquet_options=parquet_options,
                                   options=options,
                                   leveldb_write_buffer_size=leveldb_write_buffer_size,
                                   leveldb_max_file_size=leveldb_max_file_size,
                                   lmdb_map_size = lmdb_map_size,
                                   batch_size=batch_size)
        self._parallel_writer.write(data,self.input_fn, self.input_fn_args)