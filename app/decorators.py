from functools import wraps
from flask import abort
# from flask_login import current_user
# from .api.authority import Permission
import time


# def permission_required(permission):
#     def decorator(f):
#         @wraps(f)
#         def decorated_function(*args, **kwargs):
#             if not current_user.can(permission):
#                 abort(403)
#             return f(*args, **kwargs)
#         return decorated_function
#     return decorator


# def admin_required(f):
#     return permission_required(Permission.ADMIN)(f)


def time_statistics(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        start_ = time.time()
        _f = func(*args, **kwargs)
        end_ = time.time()
        time_ = end_ - start_
        print('{}的执行时间是{}'.format(func.__name__, time_))
        return _f
    return decorator

