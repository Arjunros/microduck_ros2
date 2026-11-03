#!/usr/bin/env python3
"""A ~70 s showreel for screen recording: drives the duck through walking, turning,
skills, head moves and (optionally) a natural-language command, with captions.

  terminal 1:  pixi run nav agent:=true ball:=true
  terminal 2:  start your screen recorder, then:  pixi run demo
               (pixi run demo --no-agent  to skip the language part)
"""

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"


class Demo(Node):
    def __init__(self):
        super().__init__("microduck_demo")
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.head = self.create_publisher(Float64MultiArray, "/microduck/head_cmd", 10)
        self.say_pub = self.create_publisher(String, "/microduck/command", 10)
        self.replies = []
        self.create_subscription(String, "/microduck/agent/reply", lambda m: self.replies.append(m.data), 10)

    def spin_for(self, seconds, twist=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if twist is not None:
                self.cmd.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

    def drive(self, seconds, vx=0.0, wz=0.0):
        t = Twist()
        t.linear.x, t.angular.z = float(vx), float(wz)
        self.spin_for(seconds, t)
        self.spin_for(0.3, Twist())     # stop

    def call(self, name, timeout=5.0):
        cli = self.create_client(Trigger, name)
        if not cli.wait_for_service(timeout_sec=timeout):
            print(f"{DIM}  ({name} not available, skipping){RESET}", flush=True)
            return False
        fut = cli.call_async(Trigger.Request())
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.done() and fut.result().success

    def look(self, yaw=0.0, pitch=0.0, seconds=1.2):
        self.head.publish(Float64MultiArray(data=[0.0, float(pitch), float(yaw), 0.0]))
        self.spin_for(seconds)

    def ask(self, text, timeout):
        print(f"\n   {BOLD}you>{RESET} {text}", flush=True)
        n = len(self.replies)
        self.say_pub.publish(String(data=text))
        end = time.monotonic() + timeout
        while len(self.replies) == n and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if len(self.replies) > n:
            print(f"   {BOLD}duck>{RESET} {self.replies[-1]}", flush=True)
        else:
            print(f"{DIM}   (still going, ending the recording here){RESET}", flush=True)


def caption(text, sub=""):
    print(f"\n{CYAN}{BOLD}▶ {text}{RESET}" + (f"\n  {DIM}{sub}{RESET}" if sub else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-agent", action="store_true", help="skip the natural-language part")
    ap.add_argument("--command", default="go to the kitchen", help="what to ask the agent at the end")
    ap.add_argument("--agent-timeout", type=float, default=60.0)
    ap.add_argument("--reset", action="store_true", help="reset the duck to the start pose first")
    a, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    d = Demo()
    try:
        print(f"\n{BOLD}🦆 Microduck in ROS 2{RESET}  {DIM}MuJoCo + Pollen Robotics' RL policies{RESET}", flush=True)
        if a.reset:
            d.call("/microduck/reset")
        d.spin_for(2.5)

        # The duck spawns in the corridor facing a wall ~0.5 m away (+x): turn left
        # first, then walk north up the open corridor (towards the kitchen door).
        caption("Turning in place", "Pollen's velstand RL policy, driven by /cmd_vel")
        d.drive(1.8, wz=1.0)
        caption("Walking")
        d.drive(7.0, vx=0.4)

        caption("Looking around", "head pose commands; the RViz panel is the head camera")
        d.look(yaw=0.5)
        d.look(yaw=-0.5)
        d.look(pitch=0.35)
        d.look()

        caption("Skills as ROS services", "/microduck/skill/roulade")
        d.call("/microduck/skill/roulade")
        d.spin_for(3.5)
        caption("Sit and stand", "/microduck/sit, /microduck/stand")
        d.call("/microduck/sit")
        d.spin_for(3.0)
        d.call("/microduck/stand")
        d.spin_for(4.0)

        if not a.no_agent:
            caption("Talking to it", "an LLM agent (or offline parser) plans Nav2 goals and skills")
            d.ask(a.command, a.agent_timeout)

        print(f"\n{BOLD}github.com/Arjunros/microduck_ros2{RESET}\n", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        d.cmd.publish(Twist())
        d.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
