# @Time    : 2022/11/6 10:40
import typing
import numpy as np

from transformers import BertTokenizer
from numpy_io.core.numpyadapter import ParallelNumpyWriter,NumpyReaderAdapter
# 切分词
def tokenize_data(data: typing.Any,user_data: tuple):
    tokenizer: BertTokenizer
    tokenizer,max_seq_length = user_data

    x = data
    if isinstance(x, tuple):
        o = tokenizer.encode_plus(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True,
                                  add_special_tokens=True)
    else:
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
    node = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'seqlen': input_length
    }
    return node


def make_dataset(tokenizer,data,data_backend,outputfile):
    parallel_writer = ParallelNumpyWriter(num_process_worker=0)
    parallel_writer.open(outputfile,data_backend)
    parallel_writer.write(data,tokenize_data, (tokenizer,64))



def test(tokenizer,data,data_backend,output):
    make_dataset(tokenizer,data,data_backend,output)
    dataset = NumpyReaderAdapter.load(output, data_backend)
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
    # filename = './data.txt'
    # data = DataReadLoader.read_from_file(filename)
    data = [str(i) + 'fastdatasets numpywriter demo' for i in range(1000)]

    test(tokenizer,data, 'memory_raw', [])
    test(tokenizer,data, 'memory',  [])
    test(tokenizer,data,'record','./data.record')
    test(tokenizer,data,'leveldb', './data.leveldb')
    test(tokenizer,data,'lmdb', './data.lmdb')