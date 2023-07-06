## update information
    07-06 0.0.5.post0 fix fastdatasets>=0.9.14


## numpy 序列化反序列化

## 存储

目前支持加载的权重：
- <strong>tfrecord</strong> 支持压缩 ， 支持多进程 , numpywriter 序列化反序列化
- <strong>leveldb</strong> 支持压缩 不支持多进程 , numpywriter 序列化反序列化
- <strong>lmdb</strong> 不支持压缩 , 支持多进程 , numpywriter 序列化反序列化
- <strong>memory</strong> 内存读写器, 支持多进程, numpywriter 序列化反序列化
- <strong>memory_raw</strong> 内存读写器 , 支持多进程, 原始数据迭代器，内存copy形式
- <strong>arrow_stream</strong> arrow 读写 
- <strong>arrow_file</strong> arrow 读写 
- <strong>parquet</strong> parquet 读写 
- 
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

## 适配并行读写

```text
    auto_parallel_writer.py
```