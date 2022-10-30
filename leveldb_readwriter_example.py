from tqdm import tqdm
from fastdatasets.writer.leveldb import LEVELDB_writer
from fastdatasets.leveldb_dataset import LEVELDB, load_dataset

db_path = 'd:\\example_leveldb'


def test_write(db_path):
    options = LEVELDB.LeveldbOptions(create_if_missing=True, error_if_exists=False)
    f = LEVELDB_writer(db_path, options=options)

    n = 0
    for i in range(30):
        f.put('input{}'.format(i).encode(encoding='utf-8'), str(i))
        f.put('label{}'.format(i).encode(), str(i))
        n += 1
    f.put('total_num', str(n))
    f.close()


def test_iterable(db_path):
    options = LEVELDB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = load_dataset.IterableDataset(db_path, options=options)
    for d in dataset:
        print(d)


def test_random(db_path):
    options = LEVELDB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = load_dataset.RandomDataset(db_path,
                                         data_key_prefix_list=('input', 'label'),
                                         num_key='total_num',
                                         options=options)

    dataset = dataset.shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d = dataset[i]
        print(i, d)


test_write(db_path)
test_iterable(db_path)
test_random(db_path)