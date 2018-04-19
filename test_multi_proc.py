import multiprocessing as mp
import time
from utils import fb

def error_handler(err):
    print("Error in basic task", err)
    raise err

pool = mp.Pool(300)


def sim():
    print("sim")
    try:
        reader, writer = mp.Pipe()
        results = []
        energy = 300
        print("sim1")
        while energy > 0:
            r = pool.apply_async(fb, (writer,), error_callback=error_handler)
            results.append(r)
            energy -= 1
        print("sim2")
        for i in range(300):
            r = reader.recv()
            print(r)
        print("sim3")
        [r.wait() for r in results]
    except Exception as e:
        print(e)


if __name__ == "__main__":
    p = mp.Process(target=sim)
    print("OK")
    p.start()
    p.join()


    # results = []
    # reader, writer = mp.Pipe()
    # for i in range(300):
    #     r = pool.apply_async(faaaa, (writer,i), error_callback=error_handler)
    #     results.append(r)
    # for i in range(300):
    #     print(reader.recv())
    # [r.wait() for r in results]
    # print("#########done sim")

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
