import multiprocessing as mp
import time

def f():
    print("process start")
    # time.sleep(10)
    print("process end")
    queue.put(1)
    return

queue = mp.Queue()

if __name__ == "__main__":
    # a = []
    # pool = mp.Pool(10)
    # print("active processes", len(mp.active_children()))
    # results = []
    # for i in range(30):
    #     print(i)
    #     p = mp.Process(target=f)
    #     p.start()
    #     results.append(p)
    # for i in range(len(results)):
    #     item = queue.get()
    #     a.append(item)
    #     print("a ", a)
    # [p.join() for p in results]


    import multiprocessing as mp
    import numpy as np
    from time import time
    import sys
    import fmq

    q1 = mp.Queue(10)
    q2 = fmq.Queue(10)

    # uncomment thie line to switch the order
    # q1, q2 = q2, q1

    a = np.zeros((100, 256, 256, 3))
    a_size = sys.getsizeof(a)
    print('Object size: %d bytes = %dKB = %dMB' % (a_size, a_size / 1024, a_size / 1024 / 1024))

    for i in range(10):
        q1.put(np.array(a))
        q2.put(np.array(a))

    # mp queue get
    for i in range(5):
        st = time()
        b = q1.get()
        print('mp get() a time', time() - st)

    # fmq queue get
    for i in range(5):
        st = time()
        b = q2.get()
        print('fmq get() a time', time() - st)
