"""Small Windows external sampler; stdlib only, no CUDA/NVML context or model.

Usage (construct before launching the target to keep PDH setup out of its run):
    with MemoryProbe(interval_s=0.05) as memory:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        memory.start(process.pid)
        exit_code = process.wait()
        memory.stop()
        result = memory.result()       # caller decides how to report the samples

For manual sampling use attach(pid), then sample(). Missing GPU instances are
unavailable, never zero. GPU peaks are sampled lower bounds. Per-process GPU
usage includes cross-process shared allocations; do not sum PIDs into card use.
"""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import os
import threading
import time


class _MemoryCounters(c.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
        (name, c.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


class _CounterUnion(c.Union):
    _fields_ = [("longValue", w.LONG), ("doubleValue", c.c_double),
                ("largeValue", c.c_longlong), ("stringValue", c.c_void_p)]


class _CounterValue(c.Structure):
    _fields_ = [("status", w.DWORD), ("value", _CounterUnion)]


class _CounterItem(c.Structure):
    _fields_ = [("name", c.c_wchar_p), ("formatted", _CounterValue)]


def _status(value):
    return f"0x{int(value) & 0xffffffff:08x}"


class MemoryProbe:
    """One process handle, one persistent PDH query, optional sampling thread."""

    def __init__(self, interval_s=0.05):
        if os.name != "nt":
            raise OSError("MemoryProbe requires Windows PSAPI/PDH")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)
        self.pid = None
        self.samples = []
        self._process = None
        self._query = w.HANDLE()
        self._counters = {}
        self._buffers = {}
        self._thread = None
        self._stop = threading.Event()
        self._closed = False
        self._stopped = False
        self._highwater = {}
        self._origin_ns = time.perf_counter_ns()
        self.gpu_setup_error = None
        self.thread_error = None
        self._kernel = c.WinDLL("kernel32", use_last_error=True)
        self._kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self._kernel.OpenProcess.restype = w.HANDLE
        self._kernel.CloseHandle.argtypes = [w.HANDLE]
        self._kernel.CloseHandle.restype = w.BOOL
        self._kernel.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
        self._kernel.GetExitCodeProcess.restype = w.BOOL
        self._kernel.K32GetProcessMemoryInfo.argtypes = [
            w.HANDLE, c.POINTER(_MemoryCounters), w.DWORD]
        self._kernel.K32GetProcessMemoryInfo.restype = w.BOOL
        self._pdh = c.WinDLL("pdh", use_last_error=True)
        self._pdh.PdhOpenQueryW.argtypes = [c.c_wchar_p, c.c_size_t, c.POINTER(w.HANDLE)]
        self._pdh.PdhOpenQueryW.restype = w.LONG
        self._pdh.PdhAddEnglishCounterW.argtypes = [
            w.HANDLE, c.c_wchar_p, c.c_size_t, c.POINTER(w.HANDLE)]
        self._pdh.PdhAddEnglishCounterW.restype = w.LONG
        self._pdh.PdhCollectQueryData.argtypes = [w.HANDLE]
        self._pdh.PdhCollectQueryData.restype = w.LONG
        self._pdh.PdhGetFormattedCounterArrayW.argtypes = [
            w.HANDLE, w.DWORD, c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.c_void_p]
        self._pdh.PdhGetFormattedCounterArrayW.restype = w.LONG
        self._pdh.PdhCloseQuery.argtypes = [w.HANDLE]
        self._pdh.PdhCloseQuery.restype = w.LONG
        status = self._pdh.PdhOpenQueryW(None, 0, c.byref(self._query))
        if status:
            self.gpu_setup_error = "PdhOpenQueryW " + _status(status)
        else:
            for key, name in (("dedicated", "Dedicated Usage"), ("shared", "Shared Usage")):
                handle = w.HANDLE()
                path = "\\GPU Process Memory(*)\\" + name
                status = self._pdh.PdhAddEnglishCounterW(self._query, path, 0, c.byref(handle))
                if status:
                    self.gpu_setup_error = "PdhAddEnglishCounterW " + _status(status)
                    break
                self._counters[key] = handle
            # Prime the persistent query once, before model startup. Formatting
            # the wildcard array after each collection discovers current instances.
            if not self.gpu_setup_error:
                status = self._pdh.PdhCollectQueryData(self._query)
                if status:
                    self.gpu_setup_error = "initial PdhCollectQueryData " + _status(status)
        self.setup_duration_ns = time.perf_counter_ns() - self._origin_ns

    def attach(self, pid):
        if self._closed or self.pid is not None:
            raise RuntimeError("Attach exactly once to an open MemoryProbe")
        if int(pid) <= 0:
            raise ValueError("pid must be positive")
        handle = self._kernel.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            raise c.WinError(c.get_last_error())
        self._process = handle
        self.pid = int(pid)
        self._origin_ns = time.perf_counter_ns()
        return self

    def _gpu_array(self, key):
        handle = self._counters[key]
        buffer = self._buffers.get(key)
        # A later-created GPU process/adapter instance can enlarge the array.
        # On MORE_DATA obtain a fresh zero-size measurement as documented.
        for _ in range(3):
            size = w.DWORD(len(buffer) if buffer is not None else 0)
            count = w.DWORD()
            status = self._pdh.PdhGetFormattedCounterArrayW(
                handle, 0x400, c.byref(size), c.byref(count), buffer)  # FMT_LARGE, bytes
            if (int(status) & 0xffffffff) in (0x800007d2, 0xc0000bbd):
                needed = w.DWORD()
                ignored = w.DWORD()
                status = self._pdh.PdhGetFormattedCounterArrayW(
                    handle, 0x400, c.byref(needed), c.byref(ignored), None)
                if (int(status) & 0xffffffff) != 0x800007d2:
                    return None, "counter_size " + _status(status)
                buffer = c.create_string_buffer(needed.value + 4096)
                self._buffers[key] = buffer
                continue
            if status:
                return None, "counter_array " + _status(status)
            pointer = c.cast(buffer, c.POINTER(_CounterItem))
            prefix = f"pid_{self.pid}_"
            found = {}
            invalid = False
            for i in range(count.value):
                item = pointer[i]
                name = item.name.lower()
                if not name.startswith(prefix):
                    continue
                valid = item.formatted.status in (0, 1) and item.formatted.value.largeValue >= 0
                found[name] = {
                    "status": int(item.formatted.status),
                    "bytes": int(item.formatted.value.largeValue) if valid else None,
                }
                invalid |= not valid
            if not found:
                return None, "no_pid_instance"
            return found, "invalid_pid_instance" if invalid else None
        return None, "counter_array_changed_repeatedly"

    def sample(self):
        if self.pid is None or self._closed:
            raise RuntimeError("sample requires an attached open process")
        if self._thread is not None and threading.current_thread() is not self._thread:
            raise RuntimeError("Do not call sample manually while the sampling thread is running")
        start = time.perf_counter_ns()
        row = {"timestamp_ns": start, "unix_time_ns": time.time_ns(),
               "elapsed_ns": start - self._origin_ns, "pid": self.pid}
        exit_code = w.DWORD()
        if self._kernel.GetExitCodeProcess(self._process, c.byref(exit_code)):
            row["process_running"] = exit_code.value == 259
            row["process_exit_code"] = int(exit_code.value)
        counters = _MemoryCounters()
        counters.cb = c.sizeof(counters)
        if self._kernel.K32GetProcessMemoryInfo(self._process, c.byref(counters), c.sizeof(counters)):
            row.update(psapi_available=True, working_set_bytes=int(counters.WorkingSetSize),
                       os_peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                       private_commit_bytes=int(counters.PrivateUsage),
                       os_peak_commit_bytes=int(counters.PeakPagefileUsage))
        else:
            row.update(psapi_available=False, psapi_error=c.get_last_error())
        row["gpu_instances"] = {}
        error = self.gpu_setup_error
        if not error:
            status = self._pdh.PdhCollectQueryData(self._query)
            if status:
                error = "PdhCollectQueryData " + _status(status)
        for key in ("dedicated", "shared"):
            values, reason = (None, error) if error else self._gpu_array(key)
            row[f"gpu_{key}_available"] = values is not None and reason is None
            row[f"gpu_{key}_bytes"] = (
                sum(item["bytes"] for item in values.values()) if values is not None and reason is None else None)
            if reason:
                row[f"gpu_{key}_unavailable_reason"] = reason
            for name, value in (values or {}).items():
                row["gpu_instances"].setdefault(name, {})[key] = value
        row["gpu_total_bytes"] = (
            row["gpu_dedicated_bytes"] + row["gpu_shared_bytes"]
            if row["gpu_dedicated_available"] and row["gpu_shared_available"] else None)
        for key in ("working_set_bytes", "os_peak_working_set_bytes", "private_commit_bytes",
                    "os_peak_commit_bytes", "gpu_dedicated_bytes", "gpu_shared_bytes", "gpu_total_bytes"):
            if row.get(key) is not None:
                self._highwater[key] = max(self._highwater.get(key, 0), row[key])
        row["observed_monotonic_peaks_bytes"] = dict(self._highwater)
        row["sample_duration_ns"] = time.perf_counter_ns() - start
        self.samples.append(row)
        return row

    def start(self, pid):
        self.attach(pid)
        self._thread = threading.Thread(target=self._run, name="memory-sample", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        deadline = time.perf_counter()
        try:
            while not self._stop.is_set():
                self.sample()
                deadline = max(deadline + self.interval_s, time.perf_counter())
                self._stop.wait(max(0, deadline - time.perf_counter()))
        except Exception as error:
            self.thread_error = repr(error)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        if self.pid is not None and not self._closed:
            self.sample()  # best effort after target wait; unavailable remains explicit

    def result(self):
        if self._thread is not None:
            raise RuntimeError("Stop sampling before taking the final result")
        def peak(key):
            values = [row[key] for row in self.samples if row.get(key) is not None]
            return max(values) if values else None
        peaks = {
            "sampled_max_working_set_bytes": peak("working_set_bytes"),
            "observed_os_peak_working_set_bytes": peak("os_peak_working_set_bytes"),
            "sampled_max_private_commit_bytes": peak("private_commit_bytes"),
            "observed_os_peak_commit_bytes": peak("os_peak_commit_bytes"),
            "sampled_peak_gpu_dedicated_bytes": peak("gpu_dedicated_bytes"),
            "sampled_peak_gpu_shared_bytes": peak("gpu_shared_bytes"),
            "sampled_peak_gpu_total_bytes": peak("gpu_total_bytes"),
        }
        return {"schema": "hyle.windows.process-memory-samples.v1", "pid": self.pid,
                "requested_interval_s": self.interval_s, "setup_duration_ns": self.setup_duration_ns,
                "gpu_setup_error": self.gpu_setup_error, "thread_error": self.thread_error,
                "peaks": peaks, "samples": list(self.samples),
                "limits": ["GPU peaks and current-memory maxima are sampling lower bounds; use actual timestamps.",
                           "OS peak fields are lifetime high-water marks as observed at successful PSAPI snapshots, not independent exact phase peaks.",
                           "PrivateUsage/PeakPagefileUsage measure process commit, not resident RAM or disk pagefile occupancy.",
                           "WorkingSet includes shared/mapped resident pages; CPU commit and GPU shared usage can overlap and must not be added as physical RAM.",
                           "GPU values sum this PID's reported instances across adapters; original instance LUID/physical IDs remain in every sample.",
                           "WDDM per-process GPU usage includes cross-process shared resources and cannot be summed across PIDs as card usage.",
                           "No CUDA/NVML library, context, synchronization or tensor access is used."]}

    def close(self):
        if self._closed:
            return
        self.stop()
        if self._process:
            self._kernel.CloseHandle(self._process)
            self._process = None
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = w.HANDLE()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
