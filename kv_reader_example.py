# @Time    : 2022/10/29 15:29
# @Author  : tk
# @FileName: kv_reader_example.py

from tqdm import tqdm
from fastdatasets.writer.kv_writer import DBOptions, DBIterater, DBCompressionType, DB, KV_writer
from fastdatasets import TableLoader

db_path = 'd:\\example_table2'

def test_iterable(db_path):
    options = DBOptions(create_if_missing=False, error_if_exists=False)
    dataset = TableLoader.IterableDataset(db_path, options=options)
    for d in dataset:
        print(d)


def test_random(db_path):
    options = DBOptions(create_if_missing=False, error_if_exists=False)
    dataset = TableLoader.RandomDataset(db_path,
                                        data_key_prefix_list=('input', 'label'),
                                        num_key='total_num',
                                        options=options)

    dataset = dataset.shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d = dataset[i]
        print(i, d)


if __name__ == '__main__':
    #遍历所有
    test_iterable(db_path)

    #读取
    test_random(db_path)