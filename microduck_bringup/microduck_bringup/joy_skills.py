"""Gamepad buttons -> Microduck skill services.

Default mapping (Xbox layout as reported by the `joy` node):
    A  ground_pick     X  roulade       Y  sit / stand toggle
    B  kick_right      RB kick_left     (LB is the teleop_twist_joy enable button)
Each entry is "<button index>:<service>". "sit_toggle" alternates /microduck/sit and /microduck/stand.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger

DEFAULT_MAP = ["0:/microduck/skill/ground_pick", "1:/microduck/skill/kick_right",
               "2:/microduck/skill/roulade", "3:sit_toggle", "5:/microduck/skill/kick_left"]


class JoySkills(Node):
    def __init__(self):
        super().__init__("joy_skills")
        self.declare_parameter("buttons", DEFAULT_MAP)
        self.mapping = {}
        for item in self.get_parameter("buttons").value:
            idx, _, srv = item.partition(":")
            self.mapping[int(idx)] = srv
        self.clients_ = {}
        self._prev = []
        self._state = "walk"
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.create_subscription(String, "/microduck/policy_state", self._on_state, 10)

    def _on_state(self, msg):
        self._state = msg.data

    def _client(self, name):
        if name not in self.clients_:
            self.clients_[name] = self.create_client(Trigger, name)
        return self.clients_[name]

    def _on_joy(self, msg):
        pressed = [b and not (i < len(self._prev) and self._prev[i]) for i, b in enumerate(msg.buttons)]
        self._prev = list(msg.buttons)
        for i, down in enumerate(pressed):
            if not down or i not in self.mapping:
                continue
            srv = self.mapping[i]
            if srv == "sit_toggle":
                srv = "/microduck/stand" if self._state in ("sit", "rise") else "/microduck/sit"
            cli = self._client(srv)
            if not cli.service_is_ready():
                self.get_logger().warn(f"{srv} not available")
                continue
            fut = cli.call_async(Trigger.Request())
            fut.add_done_callback(lambda f, s=srv: self._done(s, f))

    def _done(self, srv, fut):
        res = fut.result()
        if res is not None and not res.success:
            self.get_logger().warn(f"{srv}: {res.message}")


def main(args=None):
    rclpy.init(args=args)
    node = JoySkills()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
