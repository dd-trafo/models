import time


class TimeIt():
    def __init__(self, message):
        print(f'{message}... ', end='')

    def __enter__(self):
        self.start = time.time()
        return None

    def __exit__(self, type, value, traceback):
        print(f'done [{time.time() - self.start:.2f}s]')
