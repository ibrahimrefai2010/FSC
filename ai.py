import csv
import queue
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from multiprocessing import shared_memory
from ultralytics import YOLO

# Make local Yolov5_StrongSORT_OSNet importable (if script is in project root)
repo_dir = Path(__file__).resolve().parent / "Yolov5_StrongSORT_OSNet"
if repo_dir.exists():
    sys.path.insert(0, str(repo_dir))

from boxmot import create_tracker, get_tracker_config  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this pipeline, but no CUDA device was found.")

DEVICE = "cuda:0"
REID_MATCH_THRESHOLD = 0.38
LOCAL_RETAIN_THRESHOLD = 0.80
SAME_CAMERA_RELINK_THRESHOLD = 0.50
SAME_CAMERA_RECENT_SECONDS = 8.0
REID_AMBIGUITY_MARGIN = 0.0
CROSS_CAMERA_MATCH_COOLDOWN_SECONDS = 3.0
ACTIVE_LOCAL_GID_LOCK_AGE = 0
RECENT_GID_RELINK_SECONDS = 5.0
RECENT_GID_RELINK_IOU = 0.35
RECENT_GID_RELINK_FEAT_DIST = 0.55
CAMERA_FPS_ESTIMATE = 30.0
MAX_TRACK_AGE_SECONDS = 2.0
#MAX_AGE_FRAMES = max(1, int(round(CAMERA_FPS_ESTIMATE * MAX_TRACK_AGE_SECONDS)))
MAX_AGE_FRAMES = 100

GLOBAL_STALE_SECONDS = 90.0
PROTO_EMA_ALPHA = 0.88
MIN_DWELL_SECONDS = 1.0
DEFAULT_MOVEMENT_CSV = Path(__file__).resolve().parent / "camera_movements.csv"
DEFAULT_ID_DEBUG_CSV = Path(__file__).resolve().parent / "id_assign_debug.csv"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# YOLO on GPU
yolo = YOLO("yolo26x.pt")
yolo.to(DEVICE)

# StrongSORT on same GPU
tracker = create_tracker(
    tracker_type="strongsort",
    tracker_config=get_tracker_config("strongsort"),
    reid_weights=Path("osnet_ibn_x1_0_msmt17.pt"),
    device=DEVICE,
    half=True,
    evolve_param_dict={
        "det_thresh": 0.15,       # YOLO detection threshold
        "min_conf": 0.2,          # StrongSORT min confidence gate
        "max_age": 150,           # keep tracks alive for ~5 seconds at 30 FPS
        "n_init": 1,              # confirm tracks quickly to reduce local ID fragmentation
        "iou_threshold": 0.25,    # IOU threshold for association
        "max_cos_dist": 0.25,     # slightly looser to reduce local ID resets
        "max_iou_dist": 0.7,      # allow more spatial tolerance through motion/noise
        "nn_budget": 300,         # embedding memory
        "ema_alpha": 0.92,        # faster embedding updates for crowded scenes
        "per_class": False,       # track across all classes together
        "asso_func": "giou",       # standard association method
    },
    per_class=False,
)


def yolo_to_strongsort_dets(result):
    """Convert YOLO result -> np.ndarray[N,6] = [x1,y1,x2,y2,conf,cls]."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
    cls = result.boxes.cls.detach().cpu().numpy().reshape(-1, 1)
    dets = np.hstack([xyxy, conf, cls]).astype(np.float32)
    person_mask = dets[:, 5] == 0  # class 0 = person
    return dets[person_mask]


def _normalize(feature):
    feature = np.asarray(feature, dtype=np.float32)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-12:
        return None
    return feature / norm


def _iso_utc(ts):
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _ensure_id_debug_csv(csv_path, shared_state):
    ready_val = shared_state.get("id_debug_csv_ready")
    if ready_val is not None and bool(ready_val.value):
        return

    path = Path(csv_path)
    needs_header = not path.exists() or path.stat().st_size == 0
    if needs_header:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "timestamp_utc",
                    "camera_name",
                    "local_id",
                    "global_id",
                    "reason",
                    "existing_gid",
                    "prelinked_gid",
                    "best_gid",
                    "best_dist",
                    "accept_threshold",
                ]
            )

    if ready_val is not None:
        ready_val.value = True


def _append_id_debug_row(
    shared_state,
    now_ts,
    camera_name,
    local_id,
    global_id,
    reason,
    existing_gid=None,
    prelinked_gid=None,
    best_gid=None,
    best_dist=None,
    accept_threshold=None,
):
    debug_csv_path = shared_state.get("id_debug_csv_path", str(DEFAULT_ID_DEBUG_CSV))
    _ensure_id_debug_csv(debug_csv_path, shared_state)

    best_dist_str = ""
    if best_dist is not None and np.isfinite(float(best_dist)):
        best_dist_str = f"{float(best_dist):.6f}"

    accept_threshold_str = ""
    if accept_threshold is not None and np.isfinite(float(accept_threshold)):
        accept_threshold_str = f"{float(accept_threshold):.6f}"

    with Path(debug_csv_path).open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                _iso_utc(now_ts),
                str(camera_name),
                int(local_id),
                "" if global_id is None else int(global_id),
                str(reason),
                "" if existing_gid is None else int(existing_gid),
                "" if prelinked_gid is None else int(prelinked_gid),
                "" if best_gid is None else int(best_gid),
                best_dist_str,
                accept_threshold_str,
            ]
        )


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _try_recent_gid_relink(
    bbox_xyxy,
    feature,
    now_ts,
    recent_gid_memory,
    assigned_gids_in_frame,
    locked_gids_in_camera,
):
    best_gid = None
    best_iou = 0.0

    for gid, rec in recent_gid_memory.items():
        if gid in assigned_gids_in_frame or gid in locked_gids_in_camera:
            continue

        age = float(now_ts) - float(rec["ts"])
        if age > RECENT_GID_RELINK_SECONDS:
            continue

        iou = _bbox_iou_xyxy(bbox_xyxy, rec["bbox"])
        if iou < RECENT_GID_RELINK_IOU:
            continue

        rec_feat = rec.get("feature")
        if feature is not None and rec_feat is not None:
            dist = 1.0 - float(np.dot(feature, rec_feat))
            if dist > RECENT_GID_RELINK_FEAT_DIST:
                continue

        if iou > best_iou:
            best_iou = iou
            best_gid = int(gid)

    return best_gid


def _blend_proto_vector(proto_vectors, gid, feature):
    old = _normalize(proto_vectors[gid])
    if old is None:
        proto_vectors[gid] = feature
        return

    mixed = PROTO_EMA_ALPHA * old + (1.0 - PROTO_EMA_ALPHA) * feature
    mixed = _normalize(mixed)
    proto_vectors[gid] = mixed if mixed is not None else feature


def _has_cross_camera_conflict(gid, cur_camera_idx, now_ts, presence_camera_idx, last_seen):
    prev_idx = int(presence_camera_idx[gid])
    if prev_idx < 0 or prev_idx == int(cur_camera_idx):
        return False
    return (float(now_ts) - float(last_seen[gid])) < CROSS_CAMERA_MATCH_COOLDOWN_SECONDS


def _ensure_movement_csv(csv_path, shared_state):
    ready_val = shared_state["movement_csv_ready"]
    if bool(ready_val.value):
        return

    path = Path(csv_path)
    needs_header = not path.exists() or path.stat().st_size == 0
    if needs_header:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "global_id",
                    "From Attraction",
                    "To Attraction",
                    "Entered Attraction in Time UTC",
                    "Exited Attraction in Time UTC",
                    "Duration in Attraction (seconds)",
                ]
            )

    ready_val.value = True


def _init_shared_views(shared_state):
    max_global_ids = int(shared_state["max_global_ids"])
    embed_dim = int(shared_state["embed_dim"])

    proto_vectors = np.frombuffer(shared_state["proto_vectors"], dtype=np.float32).reshape(max_global_ids, embed_dim)
    proto_valid = np.frombuffer(shared_state["proto_valid"], dtype=np.int8)
    last_seen = np.frombuffer(shared_state["last_seen"], dtype=np.float64)
    presence_camera_idx = np.frombuffer(shared_state["presence_camera_idx"], dtype=np.int32)
    presence_entered_at = np.frombuffer(shared_state["presence_entered_at"], dtype=np.float64)

    return {
        "max_global_ids": max_global_ids,
        "proto_vectors": proto_vectors,
        "proto_valid": proto_valid,
        "last_seen": last_seen,
        "presence_camera_idx": presence_camera_idx,
        "presence_entered_at": presence_entered_at,
    }


def _prune_stale(views, now_ts):
    proto_valid = views["proto_valid"]
    last_seen = views["last_seen"]
    presence_camera_idx = views["presence_camera_idx"]
    presence_entered_at = views["presence_entered_at"]

    stale_mask = (proto_valid > 0) & ((now_ts - last_seen) > GLOBAL_STALE_SECONDS)
    if not np.any(stale_mask):
        return

    proto_valid[stale_mask] = 0
    last_seen[stale_mask] = 0.0
    presence_camera_idx[stale_mask] = -1
    presence_entered_at[stale_mask] = 0.0


def _log_camera_transition(gid, camera_name, now_ts, shared_state, views, csv_path):
    cam_to_idx = shared_state["camera_name_to_idx"]
    idx_to_cam = shared_state["idx_to_camera_name"]

    cur_idx = int(cam_to_idx[camera_name])
    prev_idx = int(views["presence_camera_idx"][gid])
    entered_at = float(views["presence_entered_at"][gid])

    if prev_idx < 0:
        views["presence_camera_idx"][gid] = cur_idx
        views["presence_entered_at"][gid] = float(now_ts)
        return

    if prev_idx == cur_idx:
        return

    dwell_seconds = max(0.0, float(now_ts) - entered_at)
    if dwell_seconds >= MIN_DWELL_SECONDS:
        with Path(csv_path).open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    int(gid),
                    str(idx_to_cam[prev_idx]),
                    camera_name,
                    _iso_utc(entered_at),
                    _iso_utc(now_ts),
                    f"{dwell_seconds:.3f}",
                ]
            )

    views["presence_camera_idx"][gid] = cur_idx
    views["presence_entered_at"][gid] = float(now_ts)


def _assign_global_id(
    camera_name,
    local_id,
    feature,
    shared_state,
    shared_lock,
    local_to_global,
    views,
    assigned_gids_in_frame,
    locked_gids_in_camera,
    prelinked_gid=None,
):
    feature = _normalize(feature)
    now_ts = time.time()
    if feature is None:
        with shared_lock:
            _append_id_debug_row(
                shared_state=shared_state,
                now_ts=now_ts,
                camera_name=camera_name,
                local_id=local_id,
                global_id=None,
                reason="no_feature",
            )
        return None

    with shared_lock:
        _prune_stale(views, now_ts)
        movement_csv_path = shared_state.get("movement_csv_path", str(DEFAULT_MOVEMENT_CSV))
        _ensure_movement_csv(movement_csv_path, shared_state)

        next_global_id = shared_state["next_global_id"]
        max_global_ids = views["max_global_ids"]
        proto_vectors = views["proto_vectors"]
        proto_valid = views["proto_valid"]
        last_seen = views["last_seen"]
        presence_camera_idx = views["presence_camera_idx"]
        cur_camera_idx = int(shared_state["camera_name_to_idx"][camera_name])

        existing_gid = local_to_global.get(int(local_id))
        if existing_gid is not None and 0 < int(existing_gid) < max_global_ids and proto_valid[int(existing_gid)] > 0:
            gid = int(existing_gid)
            if gid not in assigned_gids_in_frame:
                old = _normalize(proto_vectors[gid])
                if old is None:
                    _blend_proto_vector(proto_vectors, gid, feature)
                    last_seen[gid] = now_ts
                    _log_camera_transition(gid, camera_name, now_ts, shared_state, views, movement_csv_path)
                    assigned_gids_in_frame.add(gid)
                    _append_id_debug_row(
                        shared_state=shared_state,
                        now_ts=now_ts,
                        camera_name=camera_name,
                        local_id=local_id,
                        global_id=gid,
                        reason="existing_gid_no_proto",
                        existing_gid=existing_gid,
                        prelinked_gid=prelinked_gid,
                    )
                    return gid

                old_dist = 1.0 - float(np.dot(old, feature))
                if old_dist <= LOCAL_RETAIN_THRESHOLD:
                    _blend_proto_vector(proto_vectors, gid, feature)
                    last_seen[gid] = now_ts
                    _log_camera_transition(gid, camera_name, now_ts, shared_state, views, movement_csv_path)
                    assigned_gids_in_frame.add(gid)
                    _append_id_debug_row(
                        shared_state=shared_state,
                        now_ts=now_ts,
                        camera_name=camera_name,
                        local_id=local_id,
                        global_id=gid,
                        reason="existing_gid_retain",
                        existing_gid=existing_gid,
                        prelinked_gid=prelinked_gid,
                    )
                    return gid

        if existing_gid is not None:
            local_to_global.pop(int(local_id), None)

        upper = min(int(next_global_id.value), max_global_ids)
        best_gid = None
        best_dist = float("inf")

        if upper > 1:
            candidate_slice = slice(1, upper)
            valid_mask = proto_valid[candidate_slice] > 0
            if np.any(valid_mask):
                gids = np.nonzero(valid_mask)[0] + 1
                if assigned_gids_in_frame:
                    used = np.array(list(assigned_gids_in_frame), dtype=np.int32)
                    gids = gids[~np.isin(gids, used)]

                if locked_gids_in_camera:
                    blocked = set(locked_gids_in_camera)
                    if existing_gid is not None:
                        blocked.discard(int(existing_gid))
                    if blocked:
                        blocked_arr = np.array(list(blocked), dtype=np.int32)
                        gids = gids[~np.isin(gids, blocked_arr)]

                if gids.size > 0:
                    cross_cam_recent = (
                        (presence_camera_idx[gids] >= 0)
                        & (presence_camera_idx[gids] != cur_camera_idx)
                        & ((now_ts - last_seen[gids]) < CROSS_CAMERA_MATCH_COOLDOWN_SECONDS)
                    )
                    gids = gids[~cross_cam_recent]

                if gids.size > 0:
                    candidates = proto_vectors[gids]
                    dists = 1.0 - np.dot(candidates, feature)
                    order = np.argsort(dists)
                    best_idx = int(order[0])
                    best_dist = float(dists[best_idx])
                    best_gid = int(gids[best_idx])

                    if REID_AMBIGUITY_MARGIN > 0.0 and len(order) > 1:
                        second_best_dist = float(dists[int(order[1])])
                        margin = second_best_dist - best_dist
                        if margin < REID_AMBIGUITY_MARGIN:
                            best_gid = None
                            best_dist = float("inf")

        accept_threshold = REID_MATCH_THRESHOLD
        if best_gid is not None:
            if (
                int(presence_camera_idx[best_gid]) == cur_camera_idx
                and (now_ts - float(last_seen[best_gid])) <= SAME_CAMERA_RECENT_SECONDS
            ):
                accept_threshold = max(accept_threshold, SAME_CAMERA_RELINK_THRESHOLD)

        if best_gid is None or best_dist > accept_threshold:
            gid = int(next_global_id.value)
            if gid >= max_global_ids:
                gid = best_gid if best_gid is not None else (max_global_ids - 1)
            else:
                next_global_id.value = gid + 1
            reason = "new_gid"
            if best_gid is not None:
                reason = "new_gid_best_rejected"
        else:
            gid = int(best_gid)
            reason = "matched_best_gid"

        local_to_global[int(local_id)] = gid
        if 0 < gid < max_global_ids and proto_valid[gid] > 0:
            _blend_proto_vector(proto_vectors, gid, feature)
        else:
            proto_vectors[gid] = feature
        proto_valid[gid] = 1
        last_seen[gid] = now_ts
        _log_camera_transition(gid, camera_name, now_ts, shared_state, views, movement_csv_path)
        assigned_gids_in_frame.add(gid)
        _append_id_debug_row(
            shared_state=shared_state,
            now_ts=now_ts,
            camera_name=camera_name,
            local_id=local_id,
            global_id=gid,
            reason=reason,
            existing_gid=existing_gid,
            prelinked_gid=prelinked_gid,
            best_gid=best_gid,
            best_dist=best_dist,
            accept_threshold=accept_threshold,
        )
        return gid


def tick(
    ok,
    frame,
    camera_name,
    shared_state,
    shared_lock,
    local_to_global,
    views,
    recent_gid_memory,
    display_fps=0.0,
):
    if not ok:
        raise RuntimeError("Image not detected")

    result = yolo(
        frame,
        device=DEVICE,
        half=True,
        conf=0.15,
        verbose=False,
    )[0]
    dets = yolo_to_strongsort_dets(result)

    tracks = tracker.update(dets, frame)

    feat_by_local_id = {}
    for trk in tracker.tracker.tracks:
        if not trk.is_confirmed() or trk.time_since_update >= 1:
            continue
        if not trk.features:
            continue
        feat_by_local_id[int(trk.id)] = trk.features[-1]

    vis = frame
    assigned_gids_in_frame = set()
    now_ts = time.time()

    stale_gids = [gid for gid, rec in recent_gid_memory.items() if (now_ts - float(rec["ts"])) > RECENT_GID_RELINK_SECONDS]
    for gid in stale_gids:
        recent_gid_memory.pop(gid, None)

    active_local_ids = set()
    for trk in tracker.tracker.tracks:
        if trk.is_deleted():
            continue
        if not trk.is_confirmed():
            continue
        if int(trk.time_since_update) <= ACTIVE_LOCAL_GID_LOCK_AGE:
            active_local_ids.add(int(trk.id))

    locked_gids_in_camera = {
        int(local_to_global[lid])
        for lid in active_local_ids
        if lid in local_to_global and local_to_global[lid] is not None
    }

    ordered_tracks = tracks
    if isinstance(tracks, np.ndarray) and tracks.size > 0:
        ordered_tracks = sorted(tracks, key=lambda tr: float(tr[5]), reverse=True)

    for t in ordered_tracks:
        x1, y1, x2, y2 = [int(v) for v in t[:4]]
        bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
        local_id = int(t[4])
        feature = _normalize(feat_by_local_id.get(local_id))
        prelinked_gid = None

        if local_id not in local_to_global:
            relink_gid = _try_recent_gid_relink(
                bbox_xyxy=bbox_xyxy,
                feature=feature,
                now_ts=now_ts,
                recent_gid_memory=recent_gid_memory,
                assigned_gids_in_frame=assigned_gids_in_frame,
                locked_gids_in_camera=locked_gids_in_camera,
            )
            if relink_gid is not None:
                local_to_global[local_id] = int(relink_gid)
                prelinked_gid = int(relink_gid)
        elif local_id in local_to_global and local_to_global[local_id] is not None:
            prelinked_gid = int(local_to_global[local_id])

        global_id = _assign_global_id(
            camera_name,
            local_id,
            feature,
            shared_state,
            shared_lock,
            local_to_global,
            views,
            assigned_gids_in_frame,
            locked_gids_in_camera,
            prelinked_gid,
        )
        label_id = global_id if global_id is not None else local_id

        if global_id is not None and int(global_id) > 0:
            recent_gid_memory[int(global_id)] = {
                "bbox": bbox_xyxy,
                "feature": feature,
                "ts": now_ts,
            }

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"g:{label_id}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.putText(
        vis,
        f"{camera_name} FPS: {display_fps:.1f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return vis


def worker_loop(frame_queue, shared_state, shared_lock, worker_name="tracker"):
    """Run an independent camera tracker loop in a dedicated process."""
    shared_views = _init_shared_views(shared_state)
    local_to_global = {}
    recent_gid_memory = {}

    pool_cfg = shared_state.get("frame_pools", {}).get(worker_name)
    worker_shm = None
    frame_shape = None
    slot_bytes = None
    free_slots = None

    if pool_cfg is not None:
        worker_shm = shared_memory.SharedMemory(name=pool_cfg["shm_name"])
        frame_shape = tuple(pool_cfg["shape"])
        slot_bytes = int(pool_cfg["slot_bytes"])
        free_slots = pool_cfg["free_slots"]

    out_pool_cfg = shared_state.get("output_pools", {}).get(worker_name)
    out_queue = shared_state.get("output_queues", {}).get(worker_name)
    out_shm = None
    out_shape = None
    out_slot_bytes = None
    out_free_slots = None
    if out_pool_cfg is not None:
        out_shm = shared_memory.SharedMemory(name=out_pool_cfg["shm_name"])
        out_shape = tuple(out_pool_cfg["shape"])
        out_slot_bytes = int(out_pool_cfg["slot_bytes"])
        out_free_slots = out_pool_cfg["free_slots"]

    last_fps_ts = None
    display_fps = 0.0

    try:
        while True:
            try:
                while frame_queue.qsize() > 1:
                    dropped = frame_queue.get_nowait()
                    if isinstance(dropped, tuple) and len(dropped) == 2:
                        payload = dropped[1]
                        if isinstance(payload, tuple) and len(payload) >= 2 and payload[0] == "shm" and free_slots is not None:
                            try:
                                free_slots.put_nowait(int(payload[1]))
                            except Exception:
                                pass
                item = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            ok, payload = item
            frame = payload
            slot_idx = None
            if isinstance(payload, tuple) and len(payload) >= 2 and payload[0] == "shm":
                if worker_shm is None:
                    continue
                slot_idx = int(payload[1])
                offset = slot_idx * slot_bytes
                frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=worker_shm.buf, offset=offset)

            try:
                now = time.perf_counter()
                if last_fps_ts is not None:
                    dt = now - last_fps_ts
                    if dt > 0:
                        inst_fps = 1.0 / dt
                        if display_fps <= 0.0:
                            display_fps = inst_fps
                        else:
                            display_fps = (0.9 * display_fps) + (0.1 * inst_fps)
                last_fps_ts = now

                vis = tick(
                    ok,
                    frame,
                    worker_name,
                    shared_state,
                    shared_lock,
                    local_to_global,
                    shared_views,
                    recent_gid_memory,
                    display_fps,
                )

                if out_shm is not None and out_queue is not None and out_free_slots is not None:
                    try:
                        out_slot = out_free_slots.get_nowait()
                    except queue.Empty:
                        out_slot = None
                    if out_slot is not None:
                        try:
                            out_offset = int(out_slot) * out_slot_bytes
                            out_view = np.ndarray(out_shape, dtype=np.uint8, buffer=out_shm.buf, offset=out_offset)
                            out_view[...] = vis
                            try:
                                out_queue.put_nowait(int(out_slot))
                            except queue.Full:
                                try:
                                    out_free_slots.put_nowait(int(out_slot))
                                except Exception:
                                    pass
                        except Exception:
                            try:
                                out_free_slots.put_nowait(int(out_slot))
                            except Exception:
                                pass
            except KeyboardInterrupt:
                break
            except Exception:
                pass
            finally:
                if slot_idx is not None and free_slots is not None:
                    try:
                        free_slots.put_nowait(slot_idx)
                    except Exception:
                        pass
    finally:
        if worker_shm is not None:
            try:
                worker_shm.close()
            except Exception:
                pass
        if out_shm is not None:
            try:
                out_shm.close()
            except Exception:
                pass
