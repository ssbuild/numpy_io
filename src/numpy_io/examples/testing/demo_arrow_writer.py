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
                                'text': 'str',
                                'map': 'map',
                                'map2': 'map_list'
                                },
                        with_stream=with_stream,
                        options=None)
    for i in range(2):
        data = {
            "id": list(range(i * 3,(i+ 1) * 3)),
            'text': ['asdasdasdas' + str(i) for i in range(3)],
            'map': [
                {"a": "aa1" + str(i), "b": "bb1", "c": "ccccccc"},
                {"a": "aa2", "b": "bb2", "c": "ccccccc"},
                {"a": "aa3", "b": "bb3", "c": "ccccccc"},
            ],
            'map2': [

                [
                    {"a": "11" + str(i), "b": "bb", "c": "ccccccc"},
                    {"a": "12", "b": "bb", "c": "ccccccc"},
                    {"a": "13", "b": "bb", "c": "ccccccc"},
                ],
                [
                    {"a": "21", "b": "bb", "c": "ccccccc"},
                    {"a": "22", "b": "bb", "c": "ccccccc"},
                ],
                [
                    {"a": "31", "b": "bb", "c": "ccccccc"},
                    {"a": "32", "b": "bb", "c": "ccccccc"},
                    {"a": "32", "b": "bb", "c": "ccccccc22222222222222"},
                ]
            ]
        }
        # fs.write_batch(data.keys(),data.values())
        status = fs.write_batch(data.keys(),data.values())
        assert status.ok(),status.message()


    fs.close()

def test_random():
    dataset = load_dataset.RandomDataset(path_file,with_share_memory=not with_stream)
    print('total', len(dataset))
    for i in range(len(dataset)):
        print(i,dataset[i])



def test_read_iter():
    dataset = load_dataset.IterableDataset(path_file,with_share_memory=not with_stream,batch_size=1)
    for d in dataset:
        print('iter',d)


test_write()

test_random()

test_read_iter()