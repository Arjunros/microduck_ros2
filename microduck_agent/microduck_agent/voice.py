"""Talk to the duck out loud: push-to-talk speech -> text on /microduck/command,
and the duck's replies spoken back. Experimental.

  pixi run -e voice voice            # or: ros2 run microduck_agent voice
  Press Enter, speak, press Enter again.

Speech-to-text: faster-whisper (local, CPU or GPU), model `base.en` by default
(downloaded from Hugging Face on first use). Audio: sounddevice (PortAudio).
Text-to-speech: espeak-ng / espeak / spd-say if one is installed.
"""

import argparse
import queue
import shutil
import subprocess
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

RATE = 16000


def _speak_cmd():
    for exe, args in (("espeak-ng", ["-s", "165"]), ("espeak", ["-s", "165"]), ("spd-say", ["-w"])):
        if shutil.which(exe):
            return [exe, *args]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base.en", help="faster-whisper model size/name")
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    ap.add_argument("--no-speak", action="store_true", help="don't read replies aloud")
    a, ros_args = ap.parse_known_args(argv)
    try:
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as e:   # sounddevice raises OSError when PortAudio is missing
        sys.exit(f"voice needs extra packages ({e}): use `pixi run -e voice voice`, "
                 "or install faster-whisper, sounddevice and PortAudio (sudo apt install libportaudio2)")

    print(f"loading speech model '{a.model}'...", flush=True)
    compute = {"cuda": "float16", "cpu": "int8"}.get(a.device, "default")
    stt = WhisperModel(a.model, device=a.device, compute_type=compute)
    speak = None if a.no_speak else _speak_cmd()

    rclpy.init(args=ros_args)
    node = Node("microduck_voice")
    pub = node.create_publisher(String, "/microduck/command", 10)

    def on_reply(m):
        print(f"duck: {m.data}", flush=True)
        if speak:
            subprocess.Popen([*speak, m.data], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    node.create_subscription(String, "/microduck/agent/reply", on_reply, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("Press Enter to talk, Enter again to send. Ctrl-C to quit.", flush=True)
    try:
        while True:
            input()
            chunks = queue.Queue()
            with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                callback=lambda data, *_: chunks.put(data.copy())):
                print("listening... (Enter to stop)", flush=True)
                input()
            audio = np.concatenate(list(chunks.queue) or [np.zeros((1, 1), np.float32)])[:, 0]
            if len(audio) < RATE * 0.3:
                print("(too short)", flush=True)
                continue
            segments, _ = stt.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            if not text:
                print("(didn't catch that)", flush=True)
                continue
            print(f"you: {text}", flush=True)
            pub.publish(String(data=text))
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
