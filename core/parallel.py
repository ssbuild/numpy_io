# @Time    : 2022/11/5 19:34
import random
import typing
from multiprocessing import Queue,Manager,Process

__all__ = [
    'parallel_apply',
    'ParallelStruct'
]


class ParallelStruct:
    def __init__(self,
                 num_process_worker: int = 4,
                 num_process_post_worker: int = 1,
                 input_queue_size: int = 200,
                 output_queue_size : int = 100,
                 shuffle = True,
                 desc: str='parallel'):

        self.num_process_worker = num_process_worker
        self.num_process_post_worker = num_process_post_worker
        self.input_queue_size = input_queue_size
        self.output_queue_size = output_queue_size
        self.shuffle = shuffle
        self.desc = desc

        if self.input_queue_size is None:
            self.input_queue_size = -1

        if self.output_queue_size is None:
            self.output_queue_size = -1

        if self.desc is None:
            self.desc = ''

    '''
        subprocess callback: data_input process startup
    '''

    def on_input_startup(self):
        ...

    '''
        subprocess callback: data_input process
    '''

    def on_input_process(self,x):
        return x

    '''
        subprocess callback: data_input process cleanup
    '''

    def on_input_cleanup(self):
        ...

    '''
        subprocess callback: startup
    '''

    def on_output_startup(self):
        ...

    '''
        subprocess callback:process,
    '''

    def on_output_process(self, x):
        pass

    '''
        subprocess callback: cleanup
    '''

    def on_output_cleanup(self):
        ...

    '''
        main process callback
    '''
    def on_initalize(self,data):...
    '''
        main process callback
    '''
    def on_finalize(self):...


def parallel_apply(data: typing.List, parallel_node: ParallelStruct):
    Queue_CLASS = Manager().Queue if parallel_node.num_process_worker > 0 and parallel_node.num_process_worker > 0 else Queue
    q_in =Queue_CLASS(parallel_node.input_queue_size) if parallel_node.input_queue_size > 0 else Queue_CLASS()
    q_out = Queue_CLASS(parallel_node.output_queue_size) if parallel_node.output_queue_size > 0 else Queue_CLASS()


    def produce_input(q_in: Queue, q_out: Queue,
                      startup_fn: typing.Callable,
                      process_fn: typing.Callable,
                      cleanup_fn: typing.Callable,
                      ):
        startup_fn()
        while True:
            index,x = q_in.get()
            if index is None:
                break
            res = process_fn(x)
            q_out.put((index,res))
        cleanup_fn()

    def consume_output(q_out: Queue,
                       total: int,
                           startup_fn: typing.Callable,
                           process_fn: typing.Callable,
                           cleanup_fn: typing.Callable,
                           ):

        startup_fn()
        __n__ = 0

        flag = total > 0
        while flag:
            index,x = q_out.get()
            process_fn(x)
            __n__ += 1
            if __n__ == total:
                break
        cleanup_fn()



    parallel_node.on_initalize(data)

    total = len(data)
    ids = list(range(total))
    if parallel_node.shuffle:
        random.shuffle(ids)
    try:
        from tqdm import tqdm
        ids = tqdm(ids, total=total, desc=parallel_node.desc if parallel_node.desc else 'parallel_apply')
    except:
        ...

    #生成消费都是多进程
    if parallel_node.num_process_worker > 0 and parallel_node.num_process_worker > 0:
        pools = []
        for _ in range(parallel_node.num_process_worker):
            p = Process(target=produce_input, args= (q_in,
                                                    q_out,
                                                    parallel_node.on_input_startup,
                                                    parallel_node.on_input_process,
                                                    parallel_node.on_input_cleanup,
                                                    ))
            pools.append(p)
            p.start()

        for _ in range(parallel_node.num_process_post_worker):
            p = Process(target=consume_output, args=(q_out,
                                                    total,
                                                    parallel_node.on_output_startup,
                                                    parallel_node.on_output_process,
                                                    parallel_node.on_output_cleanup
                                                    ))
            pools.append(p)
            p.start()


        for i in ids:
            q_in.put((i,data[i]))

        for _ in range(parallel_node.num_process_worker):
            q_in.put((None, None))

        for p in pools:
            p.join()
    else:
        #当前进程,弃用队列
        parallel_node.on_input_startup()
        for index in ids:
            res = parallel_node.on_input_process(data[index])
            parallel_node.on_output_process(res)
        parallel_node.on_output_cleanup()

    parallel_node.on_finalize()

    try:
        del q_in
        del q_out
    except:
        pass
