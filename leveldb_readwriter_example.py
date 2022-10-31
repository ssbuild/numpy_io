import random
from tqdm import tqdm
import numpy as np
import json
import copy
from fastdatasets.leveldb import DB,load_dataset as Loader, DataType,FeatureWriter,WriterObject

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
    options = DB.LeveldbOptions(create_if_missing=True, error_if_exists=False)
    writer = WriterObject(db_path, options=options)

    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)

    n = 0
    for i in tqdm(shuffle_idx, desc='write record'):
        example = data[i]
        for key,value in example.items():
            writer.put('{}{}'.format(key,i),value)
        n += 1

    #
    writer.file_writer.put('total_num', str(n))
    writer.close()




def test_read_random(db_path):
    options = DB.LeveldbOptions(create_if_missing=False, error_if_exists=False)
    dataset = Loader.RandomDataset(db_path,
                                         data_key_prefix_list=('image','label'),
                                         num_key='total_num',
                                         options=options)

    dataset = dataset.shuffle(-1)
    print(len(dataset))
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d : dict = dataset[i]
        image,label = d.values()
        image = np.frombuffer(image,dtype=np.int32)
        image = image.reshape((128,128))

        label= np.frombuffer(label,dtype=np.int32)
        label = label.reshape((4,))
        print(image,label)
        break


if __name__ == '__main__':
    data = get_data()
    write_data(db_path,data)
    test_read_random(db_path)