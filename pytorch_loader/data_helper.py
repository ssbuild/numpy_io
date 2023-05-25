# -*- coding: utf-8 -*-
# @Author  : tk
# @Time    : 2023/5/25 9:37
import json
import os
import typing
from ..core.writer import DataWriteHelper
from .dataloaders import load_distributed_random_sampler, load_random_sampler, load_sequential_sampler


__all__ = [
    "DataPreprocessCallback",
    "DataHelperBase",
    "load_distributed_random_sampler",
    "load_random_sampler",
    "load_sequential_sampler",
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
        return load_random_sampler(*args, **kwargs)

    def load_sequential_sampler(self,*args,**kwargs):
        if 'backend' not in kwargs:
            kwargs.update({"backend": getattr(self,'backend','record')})
        return load_sequential_sampler(*args, **kwargs)





    def make_dataset(self,outfile: typing.Union[str,list],
                     data,
                     input_fn_args: typing.Any,
                     num_process_worker: int = 0,
                     shuffle: bool=True):

        self.on_data_ready()
        fw = DataWriteHelper(self.on_data_process,
                             input_fn_args,
                             outfile,
                             getattr(self,'backend','record'),
                             num_process_worker=num_process_worker,
                             shuffle=shuffle)
        fw.save(data)
        self.on_data_finalize()


def make_dataset(data: typing.List,
               input_fn:typing.Callable[[int,typing.Any,tuple],typing.Union[typing.Dict,typing.List,typing.Tuple]],
               input_fn_args:typing.Tuple,
               outfile:str,
               backend: str,
               overwrite = False,
               num_process_worker:int = 8):

    if not os.path.exists(outfile) or overwrite:
        fw = DataWriteHelper(input_fn,input_fn_args,outfile,backend,num_process_worker)
        fw.save(data)