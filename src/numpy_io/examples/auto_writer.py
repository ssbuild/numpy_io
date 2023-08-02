# -*- coding: utf-8 -*-
# @Time    : 2022/11/10 10:35
import typing
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from numpy_io.core.numpyadapter import NumpyWriterAdapter,NumpyReaderAdapter


def convert2feature(tokenizer:BertTokenizer,data,max_seq_length):
    D = []
    for x in data:
        o = tokenizer.encode_plus(x, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)
        token_type_ids = np.asarray(o['token_type_ids'], dtype=np.int64)

        input_length = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - input_length
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            token_type_ids = np.pad(token_type_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'seqlen': input_length
        }
        D.append(d)
    return D

def make_dataset(data,data_backend,outfile):
    numpy_writer = NumpyWriterAdapter(outfile,data_backend)
    batch_keys,batch_values = [],[]
    for i,d in enumerate(tqdm(data,total=len(data))):
        batch_keys.append('input{}'.format(i))
        batch_values.append(d)
        if (i +1 ) % numpy_writer.advice_batch_buffer_size == 0:
            if numpy_writer.is_kv_writer:
                numpy_writer.writer.put_batch(batch_keys,batch_values)
            else:
                numpy_writer.writer.write_batch(batch_values)
            batch_keys.clear()
            batch_values.clear()

    if len(batch_values):
        if numpy_writer.is_kv_writer:
            numpy_writer.writer.put_batch(batch_keys, batch_values)
        else:
            numpy_writer.writer.write_batch(batch_values)
    if numpy_writer.is_kv_writer:
        numpy_writer.writer.file_writer.put('total_num',str(len(data)))
    numpy_writer.close()

def test(tokenizer,data,data_backend,outfile):
    make_dataset(data,data_backend,outfile)
    dataset = NumpyReaderAdapter.load(outfile, data_backend)
    if isinstance(dataset, typing.Iterator):
        for d in dataset:
            print(d)
            break
    else:
        for i in range(len(dataset)):
            print(dataset[i])
            break
        print('total count', len(dataset))

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    data = [str(i) + 'fastdatasets numpywriter demo' for i in range(1000)]
    data = convert2feature(tokenizer,data,64)
    test(tokenizer, data, 'memory_raw', [])
    test(tokenizer, data, 'memory', [])
    test(tokenizer,data,'record', './data.record')
    test(tokenizer,data,'leveldb', './data.leveldb')
    test(tokenizer,data,'lmdb', './data.lmdb')

