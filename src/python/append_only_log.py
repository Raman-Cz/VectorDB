import struct
import hashlib
import os
import time

LOG_HEADER_FORMAT = "<II"
LOG_RECORD_FORMAT = "<IfI"
CHECKSUM_SIZE = 32

class AppendOnlyLog:
    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.record_count = 0

    def open(self):
        if not os.path.exists(self.path):
            self.file = open(self.path, "ab")
            self.file.write(struct.pack(LOG_HEADER_FORMAT, 0, 0))
            self.file.flush()
            os.fsync(self.file.fileno())
        else:
            self.file = open(self.path, "r+b")
            self._replay()
        return self

    def _replay(self):
        self.file.seek(0)
        header = self.file.read(struct.calcsize(LOG_HEADER_FORMAT))
        if not header or len(header) < struct.calcsize(LOG_HEADER_FORMAT):
            self.file.close()
            self.file = open(self.path, "wb")
            self.file.write(struct.pack(LOG_HEADER_FORMAT, 0, 0))
            self.file.flush()
            os.fsync(self.file.fileno())
            return

        magic, count = struct.unpack(LOG_HEADER_FORMAT, header)
        records = []
        while True:
            start = self.file.tell()
            raw_checksum = self.file.read(CHECKSUM_SIZE)
            if len(raw_checksum) < CHECKSUM_SIZE:
                break
            raw_header = self.file.read(struct.calcsize(LOG_RECORD_FORMAT))
            if len(raw_header) < struct.calcsize(LOG_RECORD_FORMAT):
                break
            rec_id, timestamp, data_len = struct.unpack(LOG_RECORD_FORMAT, raw_header)
            data = self.file.read(int(data_len))
            if len(data) < data_len:
                break
            checksum = hashlib.sha256(raw_header + data).digest()
            if checksum != raw_checksum:
                break
            records.append((rec_id, timestamp, data))

        self.record_count = len(records)
        self.file.seek(0, os.SEEK_END)

    def append(self, data: bytes) -> int:
        rec_id = self.record_count
        timestamp = time.time()
        raw_header = struct.pack(LOG_RECORD_FORMAT, rec_id, timestamp, len(data))
        checksum = hashlib.sha256(raw_header + data).digest()
        self.file.write(checksum)
        self.file.write(raw_header)
        self.file.write(data)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.record_count += 1
        return rec_id

    def read_all(self):
        self.file.seek(struct.calcsize(LOG_HEADER_FORMAT))
        records = []
        while True:
            raw_checksum = self.file.read(CHECKSUM_SIZE)
            if len(raw_checksum) < CHECKSUM_SIZE:
                break
            raw_header = self.file.read(struct.calcsize(LOG_RECORD_FORMAT))
            if len(raw_header) < struct.calcsize(LOG_RECORD_FORMAT):
                break
            rec_id, timestamp, data_len = struct.unpack(LOG_RECORD_FORMAT, raw_header)
            data = self.file.read(int(data_len))
            if len(data) < data_len:
                break
            records.append((rec_id, timestamp, data))
        return records

    def close(self):
        if self.file:
            self.file.close()
