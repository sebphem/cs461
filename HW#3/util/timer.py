import time
import functools

# used my old code from a project that I made while very drunk on a train in berlin
# idk if this is from ai, so im sorry if it is

def time_it(func):
    @functools.wraps(func)
    def _w(*args, **kwargs):
        _s = time.time()
        _r = func(*args,**kwargs)
        _e = time.time()
        print(func.__name__,' time taken: ', f"{_e - _s:.1f}s")
        return _r
    return _w

def time_it_batch(func):
    @functools.wraps(func)
    def _w(*args, **kwargs):
        _s = time.time()
        _r = func(*args,**kwargs)
        _e = time.time()
        return _r, _e - _s
    return _w