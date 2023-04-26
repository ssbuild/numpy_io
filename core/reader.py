# -*- coding: utf-8 -*-
# @Time:  22:44
# @Author: tk
# @File：reader
import typing
from fastdatasets.leveldb import LEVELDB
from fastdatasets.lmdb import LMDB
from fastdatasets.record import RECORD
from .numpyadapter import NumpyReaderAdapter


def load_numpy_dataset(files: typing.Union[typing.List[str], str],
                       options: typing.Union[
                           RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions] = None,
                       data_key_prefix_list=('input',),
                       num_key='total_num',
                       cycle_length=1,
                       block_length=1,
                       backend='record',
                       with_record_iterable_dataset: bool = False,
                       with_parse_from_numpy: bool = True,
                       limit_start: typing.Optional[int] = None,
                       limit_count: typing.Optional[int] = None,
                       dataset_loader_filter_fn: typing.Callable = None,
                       ):
    dataset = NumpyReaderAdapter.load(files, backend, options,
                                      data_key_prefix_list=data_key_prefix_list,
                                      num_key=num_key,
                                      cycle_length=cycle_length,
                                      block_length=block_length,
                                      with_record_iterable_dataset=with_record_iterable_dataset,
                                      with_parse_from_numpy=with_parse_from_numpy)
    if limit_start is not None and limit_start > 0:
        dataset = dataset.skip(limit_start)
    if limit_count is not None and limit_count > 0:
        dataset = dataset.limit(limit_count)
    if dataset_loader_filter_fn is not None:
        dataset = dataset_loader_filter_fn(dataset)
    return dataset

