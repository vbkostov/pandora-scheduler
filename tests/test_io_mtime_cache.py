import os
import importlib

from pandorascheduler_rework.utils import io as io_mod


def test_read_csv_cached_mtime(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("a,b\n1,2\n3,4\n")

    # Clear internal cache to make test deterministic if previously used
    try:
        io_mod._read_csv_with_mtime.cache_clear()
    except Exception:
        pass

    df1 = io_mod.read_csv_cached(str(p))
    df2 = io_mod.read_csv_cached(str(p))
    assert df1.equals(df2)

    # Modify file contents; ensure mtime changes (some filesystems have coarse mtime)
    mtime1 = p.stat().st_mtime
    p.write_text("a,b\n5,6\n7,8\n")
    mtime2 = p.stat().st_mtime
    if mtime2 == mtime1:
        # bump mtime if it didn't change
        os.utime(str(p), (mtime1 + 1, mtime1 + 1))

    df3 = io_mod.read_csv_cached(str(p))
    assert not df3.equals(df2)
    assert df3.values.tolist() == [[5, 6], [7, 8]]


def test_read_csv_cached_missing(tmp_path):
    p = tmp_path / "nope.csv"
    try:
        io_mod._read_csv_with_mtime.cache_clear()
    except Exception:
        pass
    assert io_mod.read_csv_cached(str(p)) is None
