# @Time    : 2023/4/27 20:35
# @Author  : tk
# @FileName: dataloaders
import logging
import os
import typing

import torch
from torch.utils.data import DataLoader
from fastdatasets import memory as MEMORY
from fastdatasets.common.iterable_dataset import IterableDatasetBase
from fastdatasets.common.random_dataset import RandomDatasetBase
from fastdatasets.torch_dataset import IterableDataset
from fastdatasets.torch_dataset import IterableDataset as torch_IterableDataset, Dataset as torch_Dataset
from ..core.reader import load_numpy_dataset


def check_dataset_file(files):
    if not files:
        return None

    if isinstance(files, str):
        if not os.path.exists(files):
            return None
    else:
        #检测是否是文件list
        files_ = [f for f in files if f is not None and isinstance(f, str) and os.path.exists(f)]
        if not files_:
            #检测是否是内存list
            files = [f for f in files if f is not None and isinstance(f, list)]
            if not files:
                return None
        else:
            files = files_
    return files




def load_dataset(files: typing.Union[typing.List, str],
                 shuffle: bool = False,
                 infinite: bool = False,
                 cycle_length: int = 4,
                 block_length: int = 10,
                 num_processes: int = 1,
                 process_index: int = 0,
                 backend='record',
                 with_record_iterable_dataset: bool = False,
                 with_load_memory: bool = False,
                 with_arrow_copy_to_memory=False,
                 with_torchdataset: bool = True,
                 transform_fn: typing.Callable = None,
                 check_dataset_file_fn=None,
                 limit_start: typing.Optional[int] = None,
                 limit_count: typing.Optional[int] = None,
                 dataset_loader_filter_fn: typing.Callable = None,
                 ) -> typing.Optional[typing.Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]]:
    assert process_index <= num_processes and num_processes >= 1
    check_dataset_file_fn = check_dataset_file_fn or check_dataset_file
    files = check_dataset_file_fn(files)
    if files is None:
        return None

    dataset = load_numpy_dataset(files,
                                 cycle_length=cycle_length,
                                 block_length=block_length,
                                 with_record_iterable_dataset=with_record_iterable_dataset,
                                 with_parse_from_numpy=not with_load_memory,
                                 backend=backend,
                                 limit_start=limit_start,
                                 limit_count=limit_count,
                                 dataset_loader_filter_fn=dataset_loader_filter_fn)

    if backend.startswith('arrow') or backend.startswith('parquet'):
        with_load_memory = False
        if with_arrow_copy_to_memory:
            with_load_memory = True

    # 加载至内存
    if with_load_memory:
        logging.info('load dataset to memory...')
        if isinstance(dataset, typing.Iterator):
            raw_data = [i for i in dataset]
        else:
            raw_data = [dataset[i] for i in range(len(dataset))]

        dataset = MEMORY.load_dataset.SingleRandomDataset(raw_data)
        # 解析numpy数据
        if backend != 'memory_raw' and not backend.startswith('arrow') and not backend.startswith('parquet'):
            dataset = dataset.parse_from_numpy_writer()

    if isinstance(dataset, typing.Iterator):
        dataset: IterableDatasetBase
        if num_processes > 1:
            dataset = dataset.mutiprocess(num_processes, process_index)

        if shuffle:
            dataset = dataset.shuffle(4096)

        if infinite:
            dataset = dataset.repeat(-1)

        if transform_fn is not None:
            dataset = dataset.map(transform_fn)

        dataset_ = torch_IterableDataset(dataset) if with_torchdataset else dataset
    else:
        dataset: RandomDatasetBase
        if num_processes > 1:
            dataset = dataset.mutiprocess(num_processes, process_index)

        if shuffle:
            dataset = dataset.shuffle(-1)

        if transform_fn is not None:
            dataset = dataset.map(transform_fn)

        dataset_ = torch_Dataset(dataset) if with_torchdataset else dataset
    return dataset_





def load_distributed_random_sampler(files: typing.Union[typing.List, str],
                                    batch_size,
                                    num_processes: int = 1,
                                    process_index: int = 0,
                                    collate_fn=None,
                                    pin_memory=False,
                                    backend='record',
                                    with_load_memory: bool = False,
                                    with_torchdataset: bool = True,
                                    shuffle=True,
                                    transform_fn: typing.Callable = None,
                                    check_dataset_file_fn=None,
                                    limit_start: typing.Optional[int] = None,
                                    limit_count: typing.Optional[int] = None,
                                    dataset_loader_filter_fn: typing.Callable = None,
                                    **kwargs
                                    ):
    dataset = load_dataset(
        files, shuffle=False,
        backend=backend, with_record_iterable_dataset=False,
        with_load_memory=with_load_memory, with_torchdataset=with_torchdataset,
        transform_fn=transform_fn, check_dataset_file_fn=check_dataset_file_fn,
        limit_start=limit_start,
        limit_count=limit_count,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
    )
    if dataset is None:
        return None

    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=num_processes,
                                                              rank=process_index,
                                                              shuffle=shuffle) if num_processes > 1 else None

    if not shuffle:
        do_shuffle = False
    else:
        do_shuffle = sampler is None

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=do_shuffle,
                      sampler=sampler,
                      collate_fn=collate_fn,
                      pin_memory=pin_memory, **kwargs)





def load_random_sampler(files: typing.Union[typing.List, str],
                        batch_size,
                        collate_fn=None,
                        pin_memory=False,
                        shuffle: bool = False,
                        infinite: bool = False,
                        cycle_length: int = 4,
                        block_length: int = 10,
                        num_processes: int = 1,
                        process_index: int = 0,
                        backend='record',
                        with_record_iterable_dataset: bool = False,
                        with_load_memory: bool = False,
                        with_torchdataset: bool = True,
                        transform_fn: typing.Callable = None,
                        check_dataset_file_fn=None,
                        limit_start: typing.Optional[int] = None,
                        limit_count: typing.Optional[int] = None,
                        dataset_loader_filter_fn: typing.Callable = None,
                        **kwargs
                        ) -> typing.Optional[typing.Union[
    DataLoader, torch.utils.data.Dataset, torch.utils.data.IterableDataset, IterableDatasetBase, RandomDatasetBase]]:
    dataset = load_dataset(
        files, shuffle=shuffle, infinite=infinite, cycle_length=cycle_length,
        block_length=block_length, num_processes=num_processes, process_index=process_index,
        backend=backend, with_record_iterable_dataset=with_record_iterable_dataset,
        with_load_memory=with_load_memory, with_torchdataset=with_torchdataset,
        transform_fn=transform_fn, check_dataset_file_fn=check_dataset_file_fn,
        limit_start=limit_start,
        limit_count=limit_count,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
    )
    if dataset is None:
        return None
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False if isinstance(dataset, IterableDataset) else shuffle,
                      collate_fn=collate_fn,
                      pin_memory=pin_memory, **kwargs)


