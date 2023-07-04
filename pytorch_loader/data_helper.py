# -*- coding: utf-8 -*-
# @Author  : tk
# @Time    : 2023/5/25 9:37
import json
import logging
import os
import typing
from ..core.writer import DataWriteHelper
from .dataloaders import load_distributed_random_sampler, load_random_sampler
from .tokenizer_config_helper import *

__all__ = [
    "DataPreprocessCallback",
    "DataHelperBase",
    "load_distributed_random_sampler",
    "load_random_sampler",
    'load_tokenizer',
    'load_configure',
]

class DataPreprocessCallback(object):

    # stage 1
    def on_data_ready(self):...

    # stage 2
    def on_data_process(self, data: typing.Any, user_data: tuple):
        raise NotImplemented

    # stage 3
    def on_data_finalize(self):...


    def on_task_specific_params(self) -> typing.Dict:
        return {}

    def on_get_labels(self, files: typing.List[str]):
        if not files:
            return None, None
        label_fname = files[0]
        is_json_file = label_fname.endswith('.json')
        D = set()
        with open(label_fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r\n', '').replace('\n', '')
                if not line: continue
                if is_json_file:
                    jd = json.loads(line)
                    line = jd['label']
                D.add(line)
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label

    # 读取文件
    def on_get_corpus(self, files: typing.List[str], mode: str):
        D = []
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('\r\n', '').replace('\n', '')
                    if not line: continue
                    D.append(line)
        return D


class DataHelperBase(DataPreprocessCallback):
    backend = 'record'

    def __init__(self,backend,convert_file,cache_dir,intermediate_name,):
        self.train_files = []
        self.eval_files = []
        self.test_files = []

        self.backend = backend if backend else 'record'
        self.convert_file = convert_file
        self.intermediate_name = intermediate_name
        self.cache_dir = cache_dir


    def load_distributed_random_sampler(self,*args,**kwargs):
        if 'backend' not in kwargs:
            kwargs.update({"backend": getattr(self,'backend','record')})
        kwargs.update({
            "shuffle": True,
        })
        return load_distributed_random_sampler(*args,**kwargs)

    def load_distributed_sequential_sampler(self, *args, **kwargs):
        if 'backend' not in kwargs:
            kwargs.update({"backend": getattr(self,'backend','record')})
        kwargs.update({
            "shuffle": False,
        })
        return load_distributed_random_sampler(*args, **kwargs)

    def load_random_sampler(self,*args,**kwargs):
        if 'backend' not in kwargs:
            kwargs.update({"backend": getattr(self,'backend','record')})
        kwargs.update({
            "shuffle": True,
        })
        return load_random_sampler(*args, **kwargs)

    def load_sequential_sampler(self,*args,**kwargs):
        if 'backend' not in kwargs:
            kwargs.update({"backend": getattr(self,'backend','record')})

        kwargs.update({
            "shuffle": False,
        })
        return load_random_sampler(*args, **kwargs)





    def make_dataset(self,outfile: typing.Union[str,list],
                     data,
                     input_fn_args: typing.Any,
                     num_process_worker: int = 0,
                     shuffle: bool=True,
                     options=None,
                     parquet_options: typing.Optional = None,
                     schema: typing.Optional[typing.Dict] = None,
                     leveldb_write_buffer_size=1024 * 1024 * 512,
                     leveldb_max_file_size=10 * 1024 * 1024 * 1024,
                     lmdb_map_size=1024 * 1024 * 1024 * 150,
                     batch_size=None):

        #初始化
        self.on_data_ready()
        #创建写对象上下文
        fw = DataWriteHelper(self.on_data_process,
                             input_fn_args=input_fn_args,
                             outfile=outfile,
                             backend=getattr(self,'backend','record'),
                             num_process_worker=num_process_worker,
                             shuffle=shuffle)
        #写数据回调 on_data_process
        fw.save(data,
                options=options,
                parquet_options = parquet_options,
                schema = schema,
                leveldb_write_buffer_size = leveldb_write_buffer_size,
                leveldb_max_file_size =leveldb_max_file_size,
                lmdb_map_size = lmdb_map_size,
                batch_size = batch_size)
        #写数据完成
        self.on_data_finalize()

        # 返回制作特征数据的中间文件

    def get_intermediate_file(self, intermediate_name, mode):
        if self.backend.startswith('memory'):
            # 内存数据: list
            intermediate_output = []
            logging.info('make data {} {}...'.format(self.cache_dir,
                                                     intermediate_name + '-' + mode + '.' + self.backend))
        else:
            # 本地文件数据: 文件名
            intermediate_output = os.path.join(self.cache_dir,
                                               intermediate_name + '-' + mode + '.' + self.backend)
            logging.info('make data {}...'.format(intermediate_output))
        return intermediate_output

    def make_dataset_with_args(self, input_files,
                               mode,
                               shuffle=False,
                               num_process_worker: int = 0,
                               overwrite: bool = False,
                               mixed_data=True,
                               dupe_factor=1,
                               **dataset_args):
        '''
            mode: one of [ train , eval , test]
            shuffle: whether shuffle data
            num_process_worker: the number of mutiprocess
            overwrite: whether overwrite data
            mixed_data: Whether the mixed data
        '''
        logging.info('make_dataset {} {}...'.format(','.join(input_files), mode))
        if mode == 'train':
            contain_objs = self.train_files
        elif mode == 'eval' or mode == 'val':
            contain_objs = self.eval_files
        elif mode == 'test' or mode == 'predict':
            contain_objs = self.test_files
        else:
            raise ValueError('{} invalid '.format(mode))

        if not input_files:
            logging.info('input_files empty!')
            return

        for i in range(dupe_factor):
            if self.convert_file:
                if mixed_data:
                    intermediate_name = self.intermediate_name + '_dupe_factor_{}'.format(i)
                    intermediate_output = self.get_intermediate_file(intermediate_name, mode)

                    if isinstance(intermediate_output, list) or not os.path.exists(intermediate_output) or overwrite:
                        data = self.on_get_corpus(input_files, mode)
                        self.make_dataset(intermediate_output,
                                          data,
                                          mode,
                                          num_process_worker=num_process_worker,
                                          shuffle=shuffle,
                                          **dataset_args)
                    contain_objs.append(intermediate_output)
                else:
                    for fid, input_item in enumerate(input_files):
                        intermediate_name = self.intermediate_name + '_file_{}_dupe_factor_{}'.format(fid, i)
                        intermediate_output = self.get_intermediate_file(intermediate_name, mode)

                        if isinstance(intermediate_output, list) or not os.path.exists(
                                intermediate_output) or overwrite:
                            data = self.on_get_corpus([input_item], mode)
                            self.make_dataset(intermediate_output,
                                              data,
                                              mode,
                                              num_process_worker=num_process_worker,
                                              shuffle=shuffle,
                                              **dataset_args)
                        contain_objs.append(intermediate_output)

            else:
                for input_item in input_files:
                    contain_objs.append(input_item)





















def make_dataset(data: typing.List,
               input_fn:typing.Callable[[int,typing.Any,tuple],typing.Union[typing.Dict,typing.List,typing.Tuple]],
               input_fn_args:typing.Tuple,
               outfile:str,
               backend: str,
               overwrite = False,
               num_process_worker:int = 8,
               options=None,
               parquet_options=None,
               schema=None,
               leveldb_write_buffer_size=None,
               leveldb_max_file_size=None,
               lmdb_map_size=None,
               batch_size=None
                 ):

    if not os.path.exists(outfile) or overwrite:
        fw = DataWriteHelper(input_fn,input_fn_args,outfile,backend,num_process_worker)
        fw.save(data,
                options=options,
                parquet_options=parquet_options,
                schema=schema,
                leveldb_write_buffer_size=leveldb_write_buffer_size,
                leveldb_max_file_size=leveldb_max_file_size,
                lmdb_map_size=lmdb_map_size,
                batch_size=batch_size
                )