#!/usr/bin/env python3
"""Example IMU client – simulates 6 IMU sensors and streams data to stream_server.py.

Usage:
    python examples/send_imu.py                         # connect to localhost:9000
    python examples/send_imu.py --host 192.168.1.10     # remote server
    python examples/send_imu.py --host 10.0.0.5 --port 9000 --fps 30

IMU data protocol (one JSON line per timestep, newline-terminated):
    {"t": <timestamp_s>, "imus": [[ax, ay, az, roll, pitch, yaw], ...6 entries...]}

Special commands (same TCP connection):
    {"cmd": "calibrate"}   – trigger T-pose calibration on server
    {"cmd": "reset"}       – reset LSTM state and calibration

IMU coordinate system (raw sensor frame):
    X-up  |  Y-left  |  Z-front
Accelerations in m/s²; orientations in degrees (Euler ZYX).

The server applies the IMU→model coordinate transform internally.
"""

import argparse
import json
import math
import socket
import time


def simulate_imu_frame(t: float, fps: float):
    """Generate a plausible synthetic IMU frame for testing.

    Produces a slow walking-like motion: sinusoidal torso lean + arm swing.
    6 IMUs: [pelvis, left_wrist, right_wrist, left_ankle, right_ankle, head]
    """
    omega = 2 * math.pi * 0.5     # 0.5 Hz body sway
    arm   = 2 * math.pi * 0.8     # arm swing

    g = 9.81  # gravity (m/s²)

    frames = []
    for i in range(6):
        phase = i * math.pi / 3   # stagger sensor phases

        # Gravity-dominated acceleration (sensor in roughly vertical orientation)
        ax = 0.1 * math.sin(omega * t + phase)
        ay = g + 0.05 * math.sin(omega * t * 2 + phase)
        az = 0.05 * math.cos(omega * t + phase)

        # Orientation: small oscillating angles (degrees)
        roll  = 5 * math.sin(arm * t + phase)
        pitch = 3 * math.cos(omega * t + phase)
        yaw   = 2 * math.sin(omega * t * 0.3 + phase)

        frames.append([
            round(ax, 4), round(ay, 4), round(az, 4),
            round(roll, 4), round(pitch, 4), round(yaw, 4),
        ])

    return {"t": round(t, 4), "imus": frames}


def send_command(sock: socket.socket, cmd: str):
    msg = json.dumps({"cmd": cmd}) + "\n"
    sock.sendall(msg.encode())
    print(f"[cmd] {cmd}")


def main():
    parser = argparse.ArgumentParser(description='Simulated IMU client for FIP stream server')
    parser.add_argument('--host', default='127.0.0.1', help='Server IP address')
    parser.add_argument('--port', default=9000, type=int, help='Server TCP port')
    parser.add_argument('--fps',  default=30,   type=float, help='Frames per second to send')
    parser.add_argument('--duration', default=0, type=float,
                        help='Run duration in seconds (0 = run forever)')
    parser.add_argument('--calibrate-at', default=2.0, type=float,
                        help='Send calibrate command after this many seconds (0 = skip)')
    args = parser.parse_args()

    interval = 1.0 / args.fps
    print(f"Connecting to {args.host}:{args.port} …")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((args.host, args.port))
        print(f"Connected. Sending at {args.fps} fps.")

        if args.calibrate_at > 0:
            print(f"Will send T-pose calibrate command after {args.calibrate_at}s.")

        t_start = time.monotonic()
        t_sim = 0.0
        calibrate_sent = False
        frame_count = 0

        try:
            while True:
                t_now = time.monotonic() - t_start

                if args.duration > 0 and t_now >= args.duration:
                    print(f"Duration {args.duration}s reached. Done.")
                    break

                # Send calibration command once
                if args.calibrate_at > 0 and not calibrate_sent and t_now >= args.calibrate_at:
                    send_command(sock, 'calibrate')
                    calibrate_sent = True

                # Build and send IMU frame
                frame = simulate_imu_frame(t_sim, args.fps)
                line  = json.dumps(frame) + "\n"
                sock.sendall(line.encode())

                frame_count += 1
                t_sim += interval

                if frame_count % (int(args.fps) * 5) == 0:
                    print(f"  sent {frame_count} frames ({t_now:.1f}s elapsed)")

                # Pace to target FPS
                next_send = t_start + t_sim
                sleep_for = next_send - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print(f"\nStopped after {frame_count} frames.")
        except BrokenPipeError:
            print("Server closed connection.")


if __name__ == '__main__':
    main()
