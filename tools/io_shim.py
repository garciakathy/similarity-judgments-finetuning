
# tools/io_shim.py
import builtins, os, io
from functools import wraps

class PathShim:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = PathShim()
        return cls._instance

    def __init__(self):
        self.prefix_map = []  # list of (src_prefix, dst_prefix) in order
        self.placeholders = {}  # e.g., {'model_name': 'x', 'target_layer': 'y'}
        self._patch_applied = False

    def add_prefix_map(self, src_prefix: str, dst_prefix: str):
        # Normalize without trailing slash
        src = src_prefix.rstrip('/')
        dst = dst_prefix.rstrip('/')
        # Insert earlier entries first; preserve order
        self.prefix_map.append((src, dst))

    def set_placeholders(self, d: dict):
        self.placeholders = dict(d or {})

    def translate(self, pathlike):
        # Accept PathLike, string, or anything convertible to string
        if pathlike is None:
            return pathlike
        try:
            p = os.fspath(pathlike)
        except TypeError:
            p = str(pathlike)
        s = p

        # Apply placeholders like "{model_name}" if present
        if self.placeholders and ('{' in s and '}' in s):
            try:
                s = s.format(**self.placeholders)
            except Exception:
                # leave unformatted if keys missing
                pass

        # Apply prefix maps
        for src, dst in self.prefix_map:
            if s.startswith(src + "/") or s == src:
                s = dst + s[len(src):]
                break
        return s

    # ---- monkey patches ----
    def _wrap_open(self, fn):
        @wraps(fn)
        def wrapper(file, *args, **kwargs):
            return fn(self.translate(file), *args, **kwargs)
        return wrapper

    def _wrap_exists(self, fn):
        @wraps(fn)
        def wrapper(path):
            return fn(self.translate(path))
        return wrapper

    def _wrap_makedirs(self, fn):
        @wraps(fn)
        def wrapper(name, mode=0o777, exist_ok=False):
            return fn(self.translate(name), mode=mode, exist_ok=exist_ok)
        return wrapper

    def _wrap_pandas_read(self, fn):
        @wraps(fn)
        def wrapper(filepath_or_buffer, *args, **kwargs):
            return fn(self.translate(filepath_or_buffer), *args, **kwargs)
        return wrapper

    def _wrap_pandas_to(self, fn):
        @wraps(fn)
        def wrapper(self_df, path_or_buf, *args, **kwargs):
            return fn(self_df, PathShim.instance().translate(path_or_buf), *args, **kwargs)
        return wrapper

    def _wrap_numpy_load(self, fn):
        @wraps(fn)
        def wrapper(file, *args, **kwargs):
            return fn(self.translate(file), *args, **kwargs)
        return wrapper

    def _wrap_numpy_save(self, fn):
        @wraps(fn)
        def wrapper(file, arr, *args, **kwargs):
            return fn(self.translate(file), arr, *args, **kwargs)
        return wrapper

    def _wrap_torch_load(self, fn):
        @wraps(fn)
        def wrapper(f, *args, **kwargs):
            return fn(self.translate(f), *args, **kwargs)
        return wrapper

    def _wrap_torch_save(self, fn):
        @wraps(fn)
        def wrapper(obj, f, *args, **kwargs):
            return fn(obj, self.translate(f), *args, **kwargs)
        return wrapper

    def _wrap_cv2_imread(self, fn):
        @wraps(fn)
        def wrapper(filename, *args, **kwargs):
            return fn(self.translate(filename), *args, **kwargs)
        return wrapper

    def _wrap_cv2_imwrite(self, fn):
        @wraps(fn)
        def wrapper(filename, img, *args, **kwargs):
            return fn(self.translate(filename), img, *args, **kwargs)
        return wrapper

    def _wrap_matplotlib_savefig(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], (str, bytes, os.PathLike)):
                args = (self.translate(args[0]),) + args[1:]
            elif 'fname' in kwargs:
                kwargs['fname'] = self.translate(kwargs['fname'])
            return fn(*args, **kwargs)
        return wrapper

    def apply(self):
        if self._patch_applied:
            return
        # builtins.open
        builtins.open = self._wrap_open(builtins.open)
        # os.path.exists
        os.path.exists = self._wrap_exists(os.path.exists)
        # os.makedirs
        os.makedirs = self._wrap_makedirs(os.makedirs)

        # Optional: pandas
        try:
            import pandas as pd
            pd.read_csv = self._wrap_pandas_read(pd.read_csv)
            pd.read_table = self._wrap_pandas_read(pd.read_table)
            pd.DataFrame.to_csv = self._wrap_pandas_to(pd.DataFrame.to_csv)
            pd.DataFrame.to_parquet = self._wrap_pandas_to(pd.DataFrame.to_parquet)
        except Exception:
            pass

        # numpy
        try:
            import numpy as np
            np.load = self._wrap_numpy_load(np.load)
            np.save = self._wrap_numpy_save(np.save)
        except Exception:
            pass

        # torch
        try:
            import torch
            torch.load = self._wrap_torch_load(torch.load)
            torch.save = self._wrap_torch_save(torch.save)
        except Exception:
            pass

        # cv2
        try:
            import cv2
            cv2.imread = self._wrap_cv2_imread(cv2.imread)
            cv2.imwrite = self._wrap_cv2_imwrite(cv2.imwrite)
        except Exception:
            pass

        # matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.savefig = self._wrap_matplotlib_savefig(plt.savefig)
        except Exception:
            pass

        self._patch_applied = True

# Apply immediately on import so scripts don't have to call apply() manually.
PathShim.instance().apply()
