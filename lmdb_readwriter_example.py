from tqdm import tqdm
from fastdatasets.writer.lmdb import LMDB_writer
from fastdatasets.lmdb_dataset import LMDB, load_dataset

db_path = 'd:\\example_lmdb_new2'


def test_write(db_path):
    options = LMDB.LmdbOptions(env_open_flag=0,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)

    f = LMDB_writer(db_path, options=options, map_size=1024 * 1024 * 1024)

    n = 0
    for i in range(20):
        f.put('input{}'.format(i).encode(encoding='utf-8'), str(i))
        f.put('label{}'.format(i).encode(), str(i))
        n += 1
    f.put('total_num', str(n))
    f.close()


def test_iterable(db_path):
    options = LMDB.LmdbOptions(env_open_flag=LMDB.LmdbFlag.MDB_RDONLY,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)
    dataset = load_dataset.IterableDataset(db_path, options=options)
    for d in dataset:
        print(d)


def test_random(db_path):
    options = LMDB.LmdbOptions(env_open_flag=LMDB.LmdbFlag.MDB_RDONLY,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)
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