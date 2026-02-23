import os
import queue
import socket
import sys
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import cv2
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory

import ai

HOST = "0.0.0.0"
CHANNELS = 4  # RGBA8 from Unreal
WIDTH = 2048
HEIGHT = 1024
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
HALF_WIDTH = WIDTH // 2
TRACKER_FRAME_SHAPE = (HEIGHT, HALF_WIDTH, 3)
TRACKER_FRAME_BYTES = int(np.prod(TRACKER_FRAME_SHAPE))

QUEUE_SIZE = 5
FRAME_POOL_SLOTS = 8
OUTPUT_POOL_SLOTS = 4

STREAMS = {
    4040: ("Pisa", "pisa", "burj", cv2.COLOR_RGBA2RGB),
    4041: ("Center", "eiffel", "center", cv2.COLOR_RGBA2RGB),
}

TRACKER_NAMES = ["pisa", "burj", "eiffel", "center"]
CAMERA_NAME_TO_IDX = {name: idx for idx, name in enumerate(TRACKER_NAMES)}

MAX_GLOBAL_IDS = 8192
EMBED_DIM = 512
CAMERA_FPS_ESTIMATE = 30.0
DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def recv_exact_into(conn, buf_mv, size):
    view = buf_mv[:size]
    received = 0
    while received < size:
        chunk = conn.recv_into(view[received:], size - received)
        if chunk == 0:
            return False
        received += chunk
    return True


def try_enqueue_shm(frame_queues, frame_pools_runtime, name, frame_view):
    q = frame_queues.get(name)
    pool = frame_pools_runtime.get(name)
    if q is None or pool is None:
        return

    try:
        slot = pool["free_slots"].get_nowait()
    except queue.Empty:
        return

    try:
        offset = slot * pool["slot_bytes"]
        slot_view = np.ndarray(pool["shape"], dtype=np.uint8, buffer=pool["shm"].buf, offset=offset)
        slot_view[...] = frame_view
        try:
            q.put_nowait((True, ("shm", int(slot))))
        except queue.Full:
            try:
                pool["free_slots"].put_nowait(slot)
            except Exception:
                pass
    except Exception:
        try:
            pool["free_slots"].put_nowait(slot)
        except Exception:
            pass


def tcp_reader(port, _sender_name, left_target, right_target, _color_code, frame_queues, frame_pools_runtime, stop_event):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, port))
        server.listen(1)
        server.settimeout(1.0)

        while not stop_event.is_set():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with closing(conn):
                conn.settimeout(2.0)
                raw_buf = bytearray(FRAME_SIZE)
                raw_mv = memoryview(raw_buf)
                frame_rgba = np.ndarray((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8, buffer=raw_buf)

                while not stop_event.is_set():
                    try:
                        ok = recv_exact_into(conn, raw_mv, FRAME_SIZE)
                    except socket.timeout:
                        continue
                    except OSError:
                        break

                    if not ok:
                        break

                    frame_rgb = frame_rgba[:, :, :3]
                    left_frame = frame_rgb[:, :HALF_WIDTH]
                    right_frame = frame_rgb[:, HALF_WIDTH:]

                    try_enqueue_shm(frame_queues, frame_pools_runtime, left_target, left_frame)
                    try_enqueue_shm(frame_queues, frame_pools_runtime, right_target, right_frame)


def _create_pool(name, shape, slots, prefix):
    slot_bytes = int(np.prod(shape))
    total_bytes = slot_bytes * slots
    shm = shared_memory.SharedMemory(create=True, size=total_bytes, name=f"{prefix}_{name}_{os.getpid()}")
    free_slots = mp.Queue(maxsize=slots)
    for idx in range(slots):
        free_slots.put_nowait(idx)

    runtime = {
        "shm": shm,
        "free_slots": free_slots,
        "shape": shape,
        "slot_bytes": slot_bytes,
    }
    wire = {
        "shm_name": shm.name,
        "shape": shape,
        "slot_bytes": slot_bytes,
        "free_slots": free_slots,
    }
    return runtime, wire


def _make_combined_writer(frame_shape):
    DEFAULT_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DEFAULT_RECORDINGS_DIR / f"combined_{ts}.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        CAMERA_FPS_ESTIMATE,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def _build_mosaic(frames):
    top = np.hstack([frames["burj"], frames["pisa"]])
    bottom = np.hstack([frames["center"], frames["eiffel"]])
    return np.vstack([top, bottom])


def _drain_worker_output(output_queue, output_pool_runtime, latest_frame):
    updated = False
    while True:
        try:
            slot = output_queue.get_nowait()
        except queue.Empty:
            break

        try:
            offset = int(slot) * output_pool_runtime["slot_bytes"]
            view = np.ndarray(
                output_pool_runtime["shape"],
                dtype=np.uint8,
                buffer=output_pool_runtime["shm"].buf,
                offset=offset,
            )
            latest_frame[...] = view
            updated = True
        finally:
            try:
                output_pool_runtime["free_slots"].put_nowait(int(slot))
            except Exception:
                pass

    return updated


def main():
    start_method = "spawn" if sys.platform.startswith("win") else "fork"
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    frame_queues = {name: mp.Queue(maxsize=QUEUE_SIZE) for name in TRACKER_NAMES}
    output_queues = {name: mp.Queue(maxsize=QUEUE_SIZE) for name in TRACKER_NAMES}
    workers = {}

    shared_lock = mp.Lock()

    shared_state = {
        "movement_csv_path": str(Path(__file__).resolve().parent / "camera_movements.csv"),
        "movement_csv_ready": mp.Value("b", False),
        "id_debug_csv_path": str(Path(__file__).resolve().parent / "id_assign_debug.csv"),
        "id_debug_csv_ready": mp.Value("b", False),
        "next_global_id": mp.Value("i", 1),
        "max_global_ids": MAX_GLOBAL_IDS,
        "embed_dim": EMBED_DIM,
        "proto_vectors": mp.Array("f", MAX_GLOBAL_IDS * EMBED_DIM, lock=False),
        "proto_valid": mp.Array("b", MAX_GLOBAL_IDS, lock=False),
        "last_seen": mp.Array("d", MAX_GLOBAL_IDS, lock=False),
        "presence_camera_idx": mp.Array("i", MAX_GLOBAL_IDS, lock=False),
        "presence_entered_at": mp.Array("d", MAX_GLOBAL_IDS, lock=False),
        "camera_name_to_idx": CAMERA_NAME_TO_IDX,
        "idx_to_camera_name": TRACKER_NAMES,
        "frame_pools": {},
        "output_pools": {},
        "output_queues": output_queues,
    }

    presence_idx = np.frombuffer(shared_state["presence_camera_idx"], dtype=np.int32)
    presence_idx.fill(-1)

    frame_pools_runtime = {}
    frame_pools_wire = {}
    output_pools_runtime = {}
    output_pools_wire = {}

    for name in TRACKER_NAMES:
        in_runtime, in_wire = _create_pool(name, TRACKER_FRAME_SHAPE, FRAME_POOL_SLOTS, "sprout_in")
        frame_pools_runtime[name] = in_runtime
        frame_pools_wire[name] = in_wire

        out_runtime, out_wire = _create_pool(name, TRACKER_FRAME_SHAPE, OUTPUT_POOL_SLOTS, "sprout_out")
        output_pools_runtime[name] = out_runtime
        output_pools_wire[name] = out_wire

    shared_state["frame_pools"] = frame_pools_wire
    shared_state["output_pools"] = output_pools_wire

    for name in TRACKER_NAMES:
        proc = mp.Process(
            target=ai.worker_loop,
            args=(frame_queues[name], shared_state, shared_lock, name),
            daemon=True,
        )
        proc.start()
        workers[name] = proc

    stop_event = threading.Event()
    reader_threads = []

    for port, (sender_name, left_target, right_target, color_code) in STREAMS.items():
        t = threading.Thread(
            target=tcp_reader,
            args=(port, sender_name, left_target, right_target, color_code, frame_queues, frame_pools_runtime, stop_event),
            daemon=True,
        )
        t.start()
        reader_threads.append(t)

    latest_frames = {name: np.zeros(TRACKER_FRAME_SHAPE, dtype=np.uint8) for name in TRACKER_NAMES}
    has_frame = {name: False for name in TRACKER_NAMES}

    writer = None
    record_start_ts = None
    written_frames = 0

    try:
        while not stop_event.is_set():
            any_update = False
            for name in TRACKER_NAMES:
                updated = _drain_worker_output(output_queues[name], output_pools_runtime[name], latest_frames[name])
                if updated:
                    has_frame[name] = True
                    any_update = True

            if any(has_frame.values()):
                mosaic = _build_mosaic(latest_frames)
                cv2.imshow("verify-all", mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    stop_event.set()
                    break

                if writer is None:
                    writer = _make_combined_writer(mosaic.shape)
                    record_start_ts = time.perf_counter()
                    written_frames = 0

                now_ts = time.perf_counter()
                elapsed = max(0.0, now_ts - record_start_ts)
                target_frames = int(elapsed * CAMERA_FPS_ESTIMATE)
                if target_frames > written_frames:
                    while written_frames < target_frames:
                        writer.write(mosaic)
                        written_frames += 1

            if not any_update:
                if not any(t.is_alive() for t in reader_threads) and not any(p.is_alive() for p in workers.values()):
                    break
                stop_event.wait(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()

        for t in reader_threads:
            t.join(timeout=2.0)

        for name in TRACKER_NAMES:
            try:
                frame_queues[name].put(None, timeout=0.1)
            except Exception:
                pass

        for proc in workers.values():
            proc.join(timeout=2.0)

        if writer is not None:
            writer.release()

        for pool in frame_pools_runtime.values():
            try:
                pool["shm"].close()
            except Exception:
                pass
            try:
                pool["shm"].unlink()
            except Exception:
                pass

        for pool in output_pools_runtime.values():
            try:
                pool["shm"].close()
            except Exception:
                pass
            try:
                pool["shm"].unlink()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
