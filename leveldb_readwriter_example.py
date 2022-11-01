import random
from tqdm import tqdm
import numpy as np
import json
import copy
from fastdatasets.leveldb import DB,load_dataset as Loader, DataType,FeatureWriter,WriterObject,BytesWriter

db_path = 'd:\\example_leveldb'

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

def write_data(db_path,data):
    print('write_data...')
    options = DB.LeveldbOptions(create_if_missing=True, error_if_exists=False,write_buffer_size=1024 * 1024 * 512 )
    writer = BytesWriter(db_path, options=options)
    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)
    n = len(shuffle_idx)

    keys,values=[],[]
    for i in tqdm(shuffle_idx, desc='write record'):
        example = data[i]
        for key,value in example.items():
            keys.append('{}{}'.format(key,i))
            values.append(value)

        if (i + 1) % 100000 == 0:
            writer.file_writer.put_batch(keys,values)
            keys.clear()
            values.clear()

    if len(keys):
        writer.file_writer.put_batch(keys, values)

    writer.file_writer.put('total_num', str(n))
    writer.close()

def test_read_random(db_path):
    print('load data...')
    options = DB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
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