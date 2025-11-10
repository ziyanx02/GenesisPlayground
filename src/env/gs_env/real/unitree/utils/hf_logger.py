import queue
import threading
from collections import deque

import numpy as np


class AsyncHFLogger:
    """
    Async logger with a *bounded* ring buffer of size max_seconds * rate_hz.
    Real-time thread does only queue.put().
    Consumer thread drains and stores into bounded deques.
    """

    def __init__(self, nj: int, max_seconds: float = 10.0, rate_hz: int = 1000) -> None:
        self.nj = nj
        self.running = False

        # how many samples we allow
        self.maxlen = int(max_seconds * rate_hz)

        # store final results here
        self.t_buf = deque(maxlen=self.maxlen)
        self.q_buf = deque(maxlen=self.maxlen)
        self.dq_buf = deque(maxlen=self.maxlen)
        self.tau_buf = deque(maxlen=self.maxlen)

        # IPC queue between the 1 kHz producer and async consumer
        self.q = queue.SimpleQueue()

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._consumer, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.q.put(None)  # sentinel
        self.thread.join()

        # convert to numpy once
        self.t_s = np.asarray(self.t_buf, dtype=np.int64)
        self.q_arr = np.asarray(self.q_buf, dtype=np.float32)
        self.dq_arr = np.asarray(self.dq_buf, dtype=np.float32)
        self.tau_arr = np.asarray(self.tau_buf, dtype=np.float32)

    def push(self, t_ns: int, q: list[float], dq: list[float], tau: list[float]) -> None:
        # real-time thread: only queue.put()
        if self.running:
            self.q.put((t_ns, tuple(q), tuple(dq), tuple(tau)))

    def _consumer(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            t_ns, q, dq, tau = item

            # append to ring buffers (bounded size)
            self.t_buf.append(t_ns)
            self.q_buf.append(q)
            self.dq_buf.append(dq)
            self.tau_buf.append(tau)

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.t_s, self.q_arr, self.dq_arr, self.tau_arr
