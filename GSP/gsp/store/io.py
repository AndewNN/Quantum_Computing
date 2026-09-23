"""Byte-reproducible file writing: deterministic .npz, atomic writes, sha256.

`np.savez_compressed` stamps the current time into every zip entry, so two identical freezes
would produce different bytes and different CHECKSUMS. `save_npz` writes the same arrays with
a fixed timestamp and fixed attributes, so the bytes depend only on the arrays (and zlib).
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in arrays.items():
            arr = np.asanyarray(value)
            if arr.dtype == object:
                raise TypeError(f"{name}: object arrays are not allowed (no pickles in results)")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            member = io.BytesIO()
            arr = arr if arr.flags.c_contiguous else arr.copy(order="C")   # keeps 0-d arrays 0-d
            np.lib.format.write_array(member, arr, allow_pickle=False)
            zf.writestr(info, member.getvalue(), compresslevel=6)
    return buf.getvalue()


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)          # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_npz(path, arrays: dict) -> bytes:
    data = npz_bytes(arrays)
    atomic_write_bytes(path, data)
    return data


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
