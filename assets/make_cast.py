"""Generate an asciinema v2 cast file for the CheckAgent demo animation."""
import json
import subprocess
import sys


WIDTH = 100
HEIGHT = 40
TITLE = "CheckAgent — pytest-native testing for AI agents"

PROMPT = "\x1b[1;32muser@demo\x1b[0m:\x1b[1;34m~/myproject\x1b[0m$ "

events = []
t = 0.0


def emit(text, delay=0.0):
    global t
    t += delay
    events.append([round(t, 4), "o", text])


def type_command(cmd, speed=0.05):
    emit(PROMPT)
    for ch in cmd:
        emit(ch, speed)
    emit("\r\n", 0.3)


def show_output(text, line_delay=0.02):
    for line in text.splitlines():
        emit(line + "\r\n", line_delay)


# --- Act 1: checkagent demo ---
type_command("checkagent demo")

result1 = subprocess.run(
    ["checkagent", "demo"],
    capture_output=True, text=True
)
show_output(result1.stdout + result1.stderr, line_delay=0.03)

emit("", 1.5)

# --- Act 2: checkagent scan ---
type_command("checkagent scan assets/demo_agent.py:my_agent")

result2 = subprocess.run(
    ["checkagent", "scan", "assets/demo_agent.py:my_agent"],
    capture_output=True, text=True
)
# Show only through the severity table (stop before the long findings detail)
scan_out = result2.stdout + result2.stderr
clip_marker = "Findings Detail"
if clip_marker in scan_out:
    scan_out = scan_out[:scan_out.index(clip_marker)].rstrip()
show_output(scan_out, line_delay=0.02)

emit("", 2.0)

# --- Final prompt ---
emit(PROMPT, 0.3)

# Build cast file
header = {
    "version": 2,
    "width": WIDTH,
    "height": HEIGHT,
    "timestamp": 1748786400,
    "title": TITLE,
    "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
}

out_path = "assets/demo.cast"
with open(out_path, "w") as f:
    f.write(json.dumps(header) + "\n")
    for ev in events:
        f.write(json.dumps(ev) + "\n")

print(f"Cast written to {out_path} ({len(events)} events, {round(t, 1)}s total)")
