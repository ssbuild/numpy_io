# @Time    : 2022/10/27 20:37
# @Author  : tk
import numpy as np
from tqdm import tqdm
from fastdatasets.memory import MEMORY,load_dataset,WriterObject,DataType,StringWriter,FeatureWriter,NumpyWriter

db_path = []

def test_write(db_path):
    options = MEMORY.MemoryOptions()
    f = WriterObject(db_path, options = options)
    values = []
    n = 30
    for i in range(n):
        train_node = {
            "index":np.asarray(i,dtype=np.int64),
            'image': np.random.rand(3,4),
            'labels': np.random.randint(0,21128,size=(10),dtype=np.int64),
            'bdata': np.asarray(b'11111111asdadasdasdaa')
        }
        values.append(train_node)
        if (i+1) % 10000 == 0:
            f.write_batch(values)
            values.clear()
    if len(values):
        f.write_batch(values)
    f.close()


def test_random(db_path):
    options = MEMORY.MemoryOptions()
    dataset = load_dataset.RandomDataset(db_path,options = options)

    dataset = dataset.shuffle(10)
    print(len(dataset))
    for i in tqdm(range(len(dataset)),total=len(dataset)):
        d = dataset[i]
        print(i,d)

test_write(db_path)
test_random(db_path)