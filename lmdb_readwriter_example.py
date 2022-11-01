import random
from tqdm import tqdm
import numpy as np
import json
import copy
from fastdatasets.lmdb import DB,load_dataset as Loader, DataType,FeatureWriter,WriterObject,BytesWriter

db_path = 'd:\\example_lmdb'

def get_data():
    labels = np.asarray([0, 0, 0, 1],dtype=np.int32)
    one_node = {
        'image':  np.random.randint(0,256,size=(128,128),dtype=np.int32).tobytes(),
        'label': labels.tobytes()
    }
    num_gen = 100
    print('gen {} data ....'.format(num_gen))
    data = [copy.deepcopy(one_node) for i in range(num_gen)]
    return data

def write_data(db_path,data,map_size=1024 * 1024 * 1024 * 20):
    print('write_data...')
    options = DB.LmdbOptions(env_open_flag=0,
                               env_open_mode=0o664,  # 8进制表示
                               txn_flag=0,
                               dbi_flag=0,
                               put_flag=0)

    writer = BytesWriter(db_path, options=options,map_size=map_size)

    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)

    n = len(shuffle_idx)
    keys, values = [], []
    for i in tqdm(shuffle_idx, desc='write record'):
        example = data[i]
        for key, value in example.items():
            keys.append('{}{}'.format(key, i))
            values.append(value)

        if (i + 1) % 100000 == 0:
            writer.file_writer.put_batch(keys, values)
            keys.clear()
            values.clear()


    if len(keys):
        writer.file_writer.put_batch(keys, values)

    writer.file_writer.put('total_num', str(n))
    writer.close()


def test_read_random(db_path):
    print('load data...')
    options = DB.LmdbOptions(env_open_flag=DB.LmdbFlag.MDB_RDONLY,
                             env_open_mode=0o664,  # 8进制表示
                             txn_flag=0,
                             dbi_flag=0,
                             put_flag=0)
    dataset = Loader.RandomDataset(db_path,
                                         data_key_prefix_list=('image','label'),
                                         num_key='total_num',
                                         options=options)

    dataset = dataset.shuffle(-1)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset),desc='read data'):
        d : dict = dataset[i]
        image,label = d.values()
        image = np.frombuffer(image,dtype=np.int32)
        image = image.reshape((128,128))

        label= np.frombuffer(label,dtype=np.int32)
        label = label.reshape((4,))
        # print(image,label)
        # break


if __name__ == '__main__':
    data = get_data()
    write_data(db_path,data)
    test_read_random(db_path)