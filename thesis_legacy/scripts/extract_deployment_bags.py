#!/usr/bin/env python3
"""Extract deployment-relevant topics from a rosbag2 .db3 into flat CSV/NPZ.

Runs on the robot (needs autoware_* message packages on the ROS path).
Writes into <outdir>/<bagname>/.
"""
import csv
import glob
import hashlib
import math
import os
import sqlite3
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def main(bagdir, outroot):
    name = os.path.basename(bagdir.rstrip("/"))
    outdir = os.path.join(outroot, name)
    os.makedirs(outdir, exist_ok=True)

    db3 = sorted(glob.glob(os.path.join(bagdir, "*.db3")))
    assert db3, f"no .db3 in {bagdir}"

    # topic_id -> (name, type)
    rows_by_topic = {}
    for path in db3:
        con = sqlite3.connect(path)
        topics = {tid: (tname, ttype) for tid, tname, ttype in
                  con.execute("SELECT id, name, type FROM topics")}
        for tid, ts, data in con.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"):
            tname, ttype = topics[tid]
            rows_by_topic.setdefault((tname, ttype), []).append((ts, data))
        con.close()

    msg_cache = {}

    def msgcls(ttype):
        if ttype not in msg_cache:
            msg_cache[ttype] = get_message(ttype)
        return msg_cache[ttype]

    def dump(topic, header, fn):
        hits = [k for k in rows_by_topic if k[0] == topic]
        if not hits:
            print(f"  !! missing {topic}")
            return
        key = hits[0]
        cls = msgcls(key[1])
        fp = os.path.join(outdir, topic.strip("/").replace("/", "__") + ".csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ns"] + header)
            for ts, data in rows_by_topic[key]:
                m = deserialize_message(data, cls)
                w.writerow([ts] + fn(m))
        print(f"  wrote {fp} ({len(rows_by_topic[key])} rows)")

    dump("/localization/kinematic_state", ["x", "y", "yaw", "vx", "vy", "wz"],
         lambda m: [m.pose.pose.position.x, m.pose.pose.position.y,
                    yaw_from_quat(m.pose.pose.orientation),
                    m.twist.twist.linear.x, m.twist.twist.linear.y,
                    m.twist.twist.angular.z])

    dump("/vehicle/trailer_state", ["hitch_angle", "hitch_rate"],
         lambda m: [m.hitch_angle, m.hitch_rate])

    dump("/vehicle/status/steering_status", ["steering_tire_angle"],
         lambda m: [m.steering_tire_angle])

    dump("/control/command/control_cmd",
         ["steer_angle_cmd", "steer_rate_cmd", "vel_cmd", "accel_cmd"],
         lambda m: [m.lateral.steering_tire_angle,
                    m.lateral.steering_tire_rotation_rate,
                    m.longitudinal.velocity, m.longitudinal.acceleration])

    dump("/planning/lane_reference/drive_enabled", ["enabled"],
         lambda m: [int(m.data)])

    dump("/planning/lane_reference/drive_direction", ["reverse"],
         lambda m: [int(m.data)])

    for topic in ("/rl_bridge/state_vector", "/rl_bridge/raw_action"):
        hits = [k for k in rows_by_topic if k[0] == topic]
        if not hits:
            print(f"  !! missing {topic}")
            continue
        key = hits[0]
        cls = msgcls(key[1])
        recs = []
        for ts, data in rows_by_topic[key]:
            m = deserialize_message(data, cls)
            recs.append([ts] + list(m.data))
        n = max(len(r) for r in recs) - 1
        fp = os.path.join(outdir, topic.strip("/").replace("/", "__") + ".csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ns"] + [f"v{i}" for i in range(n)])
            for r in recs:
                w.writerow(r + [""] * (n + 1 - len(r)))
        print(f"  wrote {fp} ({len(recs)} rows, dim={n})")

    # Centerline: dedupe identical paths, store unique geometries + index per stamp
    hits = [k for k in rows_by_topic if k[0] == "/planning/lane_reference/centerline"]
    if hits:
        key = hits[0]
        cls = msgcls(key[1])
        uniq = {}
        order = []
        idx_rows = []
        frames = set()
        for ts, data in rows_by_topic[key]:
            m = deserialize_message(data, cls)
            frames.add(m.header.frame_id)
            pts = np.array([[p.pose.position.x, p.pose.position.y]
                            for p in m.poses], dtype=np.float64)
            h = hashlib.md5(pts.tobytes()).hexdigest()
            if h not in uniq:
                uniq[h] = len(order)
                order.append(pts)
            idx_rows.append((ts, uniq[h]))
        np.savez_compressed(os.path.join(outdir, "centerlines.npz"),
                            **{f"p{i}": p for i, p in enumerate(order)})
        with open(os.path.join(outdir, "centerline_index.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ns", "path_id"])
            w.writerows(idx_rows)
        print(f"  wrote centerlines: {len(order)} unique of {len(idx_rows)} msgs, "
              f"frames={frames}, npts={[len(p) for p in order[:5]]}")


if __name__ == "__main__":
    outroot = sys.argv[1]
    for bagdir in sys.argv[2:]:
        print(f"== {bagdir}")
        main(bagdir, outroot)
