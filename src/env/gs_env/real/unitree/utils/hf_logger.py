# hf_logger.py
import os
import struct
import threading
import time
from collections import deque
from collections.abc import Sequence

import numpy as np

MAGIC = b"HFB1"  # 4 bytes: "High-Freq Bin v1"


class SimpleHFBinLogger:
    """
    header: [MAGIC:4s][version:uint32][nj:uint32][frame_size:uint32]
    frame:  [t_ns:uint64][q:float32*nj][dq:float32*nj][tau:float32*nj]  (little-endian)
    """

    def __init__(
        self, path: str, nj: int, flush_interval_s: float = 0.01, max_queue: int = 200_000
    ) -> None:
        self.path = path + ".bin"
        self.nj = nj
        self.flush_interval_s = flush_interval_s
        self.q = deque(maxlen=max_queue)
        self.stop_evt = threading.Event()
        self.fh = None
        self._frame = struct.Struct("<Q" + "f" * (3 * nj))  # t + q + dq + tau
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._th: threading.Thread | None = None

    def start(self) -> None:
        self.fh = open(self.path, "ab", buffering=1024 * 1024)
        if self.fh.tell() == 0:
            version = 1
            frame_size = self._frame.size
            self.fh.write(struct.pack("<4sIII", MAGIC, version, self.nj, frame_size))
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self.stop_evt.set()
        if self._th:
            self._th.join()
        self._drain()
        self.fh.flush()
        self.fh.close()

    def push(
        self, t_ns: int, q: Sequence[float], dq: Sequence[float], tau: Sequence[float]
    ) -> None:
        self.q.append((t_ns, tuple(q), tuple(dq), tuple(tau)))

    def _run(self) -> None:
        buf = bytearray()
        next_flush = time.perf_counter() + self.flush_interval_s
        while not self.stop_evt.is_set():
            now = time.perf_counter()
            if now >= next_flush:
                self._drain(buf)
                next_flush = now + self.flush_interval_s
            time.sleep(0.001)
        self._drain(buf)

    def _drain(self, buf: bytearray | None = None) -> None:
        if not self.q:
            return
        if buf is None:
            buf = bytearray()
        buf.clear()
        pop = self.q.popleft
        pack = self._frame.pack
        while self.q:
            t_ns, q, dq, tau = pop()
            buf += pack(t_ns, *q, *dq, *tau)
        if buf:
            self.fh.write(buf)

    # ---------- Readers / exporters ----------

    @staticmethod
    def read_hfbin(path: str) -> dict:
        with open(path, "rb") as f:
            hdr = f.read(16)  # 4s + 3*uint32
            magic, version, nj, frame_size = struct.unpack("<4sIII", hdr)
            if magic != MAGIC:
                raise ValueError("Not an HFB1 file (bad magic).")
        dt = np.dtype(
            [
                ("t_ns", "<u8"),
                ("q", ("<f4", (nj,))),
                ("dq", ("<f4", (nj,))),
                ("tau", ("<f4", (nj,))),
            ]
        )
        data = np.fromfile(path, dtype=dt, offset=16)  # skip header
        return {"nj": nj, "version": version, "frame_size": frame_size, "data": data}

    @staticmethod
    def export_npz(bin_path: str, npz_path: str | None = None) -> str:
        if npz_path is None:
            npz_path = bin_path + ".npz"
        r = SimpleHFBinLogger.read_hfbin(bin_path)
        d = r["data"]
        np.savez(
            npz_path,
            t_ns=d["t_ns"],
            q=d["q"],
            dq=d["dq"],
            tau=d["tau"],
            nj=r["nj"],
            version=r["version"],
        )
        return npz_path
