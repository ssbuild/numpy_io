# -*- coding: utf-8 -*-
# @Time    : 2022/10/31 14:07
# -*- coding: utf-8 -*-
#python=3.6
import random
import lmdb
import numpy as np
import copy
from datetime import datetime
from tqdm import tqdm

class TimeSpan:
    def __init__(self,string=''):
        self.start(string)
    def start(self,string):
        self.string = string
        self.s = datetime.now()
        print(self.string,'..........')

    def show(self):
        e = datetime.now()
        print(self.string,': ',(e - self.s),'second: ', (e - self.s).seconds,'\n')


def get_data():
    labels = np.asarray([0, 0, 0, 1], dtype=np.int32)
    one_node = {
        'image': np.random.randint(0, 256, size=(128, 128), dtype=np.int32).tobytes(),
        'label': labels.tobytes()
    }
    num_gen = 1000000
    print('gen {} data ....'.format(num_gen))
    data = [copy.deepcopy(one_node) for i in range(num_gen)]
    return data




def test_write(db_path ,data):
    s = TimeSpan('写数数据')

    env = lmdb.open(db_path, map_size=1024 * 1024 * 1025 * 50)

    # 参数write设置为True才可以写入
    txn = env.begin(write=True)

    # 添加数据和键值

    shuffle_idx = list(range(len(data)))
    random.shuffle(shuffle_idx)
    n = 0
    for i in tqdm(shuffle_idx, desc='write record'):
        example = data[i]
        for key, value in example.items():
            txn.put('{}{}'.format(key, i).encode(), value)
        n += 1

    #
    txn.put('total_num'.encode(), str(n).encode())


    # 通过commit()函数提交更改
    txn.commit()
    env.close()
    s.show()

def lmdb_read(db_path):
    env = lmdb.Environment(db_path)
    txn = env.begin()  #write=False
    # # get函数通过键值查询数据
    total_num = int(txn.get('total_num'.encode()))

    # 通过cursor()遍历所有数据和键值
    # for key, value in tqdm(txn.cursor(),total=total_num):
    #     # print (key, value)
    #     pass

    for i in tqdm(range(total_num) ,total=total_num,desc='read record'):
        image = txn.get('image{}'.format(i).encode())
        label = txn.get('label{}'.format(i).encode())

        image = np.frombuffer(image, dtype=np.int32)
        image = image.reshape((128, 128))

        label = np.frombuffer(label, dtype=np.int32)
        label = label.reshape((4,))

    print(txn.stat())
    print(txn.stat()['entries'])  #读取LMDB文件的样本数量

    # close
    env.close()

def main():
    data = get_data()
    # lmdb_create()
    test_write('./data_lmdb',data)
    lmdb_read('./data_lmdb')



if __name__ == '__main__':
    main()
