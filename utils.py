import threading
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

import numpy as np
import random
import logging
from sklearn.metrics import accuracy_score


def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x


def random_index(n_all, n_train, seed):
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_idx = random_idx[0:n_train]
    test_idx = random_idx[n_train:n_all]
    return train_idx, test_idx


def TT_split(n_all, test_prop, seed):
    """
    split data into training, testing dataset
    """
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1-test_prop) * n_all).astype(np.int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor(test_prop * n_all).astype(np.int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx


def initLogging(logFilename):
    LOG_FORMAT = "%(asctime)s\tFile \"%(filename)s\",LINE %(lineno)-4d : %(levelname)-8s %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=logFilename, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    formatter = logging.Formatter(LOG_FORMAT);
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    console.setFormatter(formatter);
    logging.getLogger('').addHandler(console);


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


# decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)

        return bg_generator
