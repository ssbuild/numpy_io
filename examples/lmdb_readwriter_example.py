# @Time    : 2022/10/27 20:37
# @Author  : tk

import numpy as np
from tqdm import tqdm
from fastdatasets.lmdb import DB,load_dataset,WriterObject,DataType,StringWriter,JsonWriter,FeatureWriter,NumpyWriter

db_path = 'd:\\example_lmdb_numpy'

def test_write(db_path):
    options = DB.LmdbOptions(env_open_flag = 0,
                env_open_mode = 0o664, # 8进制表示
                txn_flag = 0,
                dbi_flag = 0,
                put_flag = 0)

    f = NumpyWriter(db_path, options = options,map_size=1024 * 1024 * 1024)

    # total_num = int(f.get_writer.get('total_num', 0))
    total_num = 0
    n = 30
    keys, values = [], []
    for i in range(n):
        train_node = {
            'image': np.random.rand(3, 4),
            'labels': np.random.randint(0, 21128, size=(10), dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }
        keys.append('input{}'.format(total_num+i))
        values.append(train_node)
        if (i + 1) % 10000 == 0:
            f.put_batch(keys, values)
            keys.clear()
            values.clear()
    if len(keys):
        f.put_batch(keys, values)

    f.get_writer.put('total_num', str(total_num + n))
    f.close()



def test_random(db_path):
    options = DB.LmdbOptions(env_open_flag=DB.LmdbFlag.MDB_RDONLY,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)
    dataset = load_dataset.RandomDataset(db_path,
                                        data_key_prefix_list=('input',),
                                        num_key='total_num',
                                        options = options)

    dataset = dataset.parse_from_numpy_writer().shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d = dataset[i]
        print(d)

test_write(db_path)
test_random(db_path)