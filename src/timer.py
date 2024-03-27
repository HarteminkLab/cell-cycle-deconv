
import time


class Timer:

    def __init__(self):
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.elapsed_time = time.time() - self.start_time

    def get_time(self):

        self.elapsed_time = time.time() - self.start_time

        hours, rem = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        return("{:0>2}:{:0>2}:{:06.3f}".format(int(hours),int(minutes),seconds))

    def print_time(self, str=None):
        if str is None:
            print(f"{self.get_time()}")
        else:
            print(f"{str} - {self.get_time()}")
