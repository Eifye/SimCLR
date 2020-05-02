import time

class StopWatch():

    def __init__(self):
        self._start = 0
        self._stop = 0

    def start(self):
        self._start = time.time()
        self._stop = 0
        return self._start

    def stop(self):
        self._stop = time.time()
        return self.elapsed_time

    @property
    def elapsed_time(self):
        return self._stop - self._start
    
def stopwatch(method):

    def wrapper(*args, **kwargs):
        sw = StopWatch()
        sw.start()
        ret = method(*args, **kwargs)
        elapsed = sw.stop()

        print('exec {} / {} [sec]'.format(method, elapsed))

        return ret

    return wrapper