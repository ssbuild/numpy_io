# -*- coding: utf-8 -*-
# @Time:  19:51
# @Author: tk
# @File：demo_arrow_writer


from fastdatasets.arrow.writer import PythonWriter
from fastdatasets.arrow.dataset import load_dataset,arrow


path_file = 'd:/tmp/data.arrow'


with_stream = True
def test_write():
    fs = PythonWriter(path_file,
                        schema={'id': 'int32',
                                'ids': 'int32_list',
                                'text2': 'str'
                                },
                        with_stream=with_stream,
                        options=None)
    for i in range(2):
        data = {
            "id": [0,1,2,3,4],
            'ids': [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],
            'text2': ['asdasdasdas3asdadas' + str(i) for i in range(5)]
        }
        # fs.write_batch(data.keys(),data.values())
        fs.write_table(data.keys(),data.values())


    fs.close()

def test_random():
    dataset = load_dataset.RandomDataset(path_file,with_share_memory=not with_stream)
    print('total', len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])



def test_read_iter():
    dataset = load_dataset.IterableDataset(path_file,with_share_memory=not with_stream,batch_size=1)
    for d in dataset:
        print('iter',d)


test_write()

test_random()

test_read_iter()