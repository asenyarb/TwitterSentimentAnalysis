import functools
import time


def repeat_action(secs=None):
    def _decorate(function):

        @functools.wraps(function)
        def wrapped_function(*args, **kwargs):
            while True:
                t1 = time.time()
                function(*args, **kwargs)
                time.sleep(max(0, secs - (time.time() - t1)))

        return wrapped_function

    return _decorate
