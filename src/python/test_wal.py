import os
import signal
import subprocess
import sys
import tempfile
from append_only_log import AppendOnlyLog

WAL_PATH = "/tmp/test_vector_db_wal.bin"

def test_basic_append_and_replay():
    if os.path.exists(WAL_PATH):
        os.remove(WAL_PATH)

    log = AppendOnlyLog(WAL_PATH)
    log.open()
    ids = []
    for i in range(10):
        rec_id = log.append(f"record_{i}".encode())
        ids.append(rec_id)
    log.close()
    assert ids == list(range(10)), f"Expected ids 0-9, got {ids}"

    log2 = AppendOnlyLog(WAL_PATH)
    log2.open()
    records = log2.read_all()
    assert len(records) == 10, f"Expected 10 records, got {len(records)}"
    for i, (rec_id, ts, data) in enumerate(records):
        assert data == f"record_{i}".encode(), f"Data mismatch at {i}"
    log2.close()
    os.remove(WAL_PATH)
    print("PASS: basic_append_and_replay")

def test_crash_recovery():
    if os.path.exists(WAL_PATH):
        os.remove(WAL_PATH)

    writer_script = f"""
import sys
sys.path.insert(0, 'src/python')
from append_only_log import AppendOnlyLog
log = AppendOnlyLog("{WAL_PATH}")
log.open()
for i in range(5):
    log.append(f"crash_record_{{i}}".encode())
import os
os.kill(os.getpid(), {signal.SIGKILL})
"""

    proc = subprocess.run(
        [sys.executable, "-c", writer_script],
        capture_output=True,
    )
    assert proc.returncode == -9 or proc.returncode == -signal.SIGKILL

    log = AppendOnlyLog(WAL_PATH)
    log.open()
    records = log.read_all()
    assert len(records) <= 5, f"Expected at most 5 records after crash, got {len(records)}"
    print(f"Recovered {len(records)} records after kill -9")
    for rec_id, ts, data in records:
        assert data.startswith(b"crash_record_"), f"Unexpected data: {data}"
    log.close()
    os.remove(WAL_PATH)
    print("PASS: crash_recovery")

def test_incomplete_write_detection():
    if os.path.exists(WAL_PATH):
        os.remove(WAL_PATH)

    log = AppendOnlyLog(WAL_PATH)
    log.open()
    for i in range(3):
        log.append(f"good_record_{i}".encode())

    with open(WAL_PATH, "ab") as f:
        f.write(b"garbage_incomplete_data_at_end")

    log2 = AppendOnlyLog(WAL_PATH)
    log2.open()
    records = log2.read_all()
    assert len(records) == 3, f"Expected 3 good records, got {len(records)} (incomplete write should be ignored)"
    log2.close()
    os.remove(WAL_PATH)
    print("PASS: incomplete_write_detection")

if __name__ == "__main__":
    test_basic_append_and_replay()
    test_crash_recovery()
    test_incomplete_write_detection()
    print("All WAL tests passed!")
