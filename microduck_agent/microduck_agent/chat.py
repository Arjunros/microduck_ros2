"""Type to the duck.

  ros2 run microduck_agent chat
  duck> go to the kitchen and sit
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main(args=None):
    rclpy.init(args=args)
    node = Node("microduck_chat")
    pub = node.create_publisher(String, "/microduck/command", 10)
    replied = threading.Event()
    verbose = "-v" in sys.argv

    def on_reply(m):
        print(f"\r\033[Kduck: {m.data}\nyou> ", end="", flush=True)
        replied.set()

    def on_event(m):
        if verbose and not m.data.startswith(("thinking", "idle")):
            print(f"\r\033[K  [{m.data}]\nyou> ", end="", flush=True)

    node.create_subscription(String, "/microduck/agent/reply", on_reply, 10)
    node.create_subscription(String, "/microduck/agent/events", on_event, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("Talk to Microduck (Ctrl-D to quit, 'stop' stops it at once, -v shows tool calls).")
    try:
        while True:
            line = input("you> ")
            if line.strip():
                pub.publish(String(data=line))
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
