import multiprocessing as mp
import time

def f():
    print("process start")
    # time.sleep(5)
    print("process end")
    queue.put(1)
    return

queue = mp.Queue()
pool = mp.Pool(60)
import fmq
q2 = fmq.Queue()

def sim():
    results = []
    for i in range(30):
        r = pool.apply_async(f)
        results.append(r)
    [r.wait() for r in results]


if __name__ == "__main__":
    for i in range(30):
        sim()
    print("#########done sim")

    # for i in range(len(results)):
    #     item = queue.get()
    #     a.append(item)
    #     print("a ", a)
    # [p.join() for p in results]


    # import numpy as np
    # from time import time
    # import sys
    #
    # # uncomment thie line to switch the order
    # # q1, q2 = q2, q1
    #
    # a = np.zeros((100, 256, 256, 3))
    # a_size = sys.getsizeof(a)
    # print('Object size: %d bytes = %dKB = %dMB' % (a_size, a_size / 1024, a_size / 1024 / 1024))
    #
    # for i in range(5):
    #     q2.put((np.array(a), "1", "2"))
    #
    #
    #
    # # fmq queue get
    # for i in range(5):
    #     st = time()
    #     b = q2.get()
    #     print('fmq get() a time', time() - st)
