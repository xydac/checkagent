#!/bin/bash
# Script to be run inside asciinema recording
# Produces a clean demo: checkagent demo + checkagent scan

export PS1='$ '
export PYTHONDONTWRITEBYTECODE=1
cd /home/x/working/agentprobe/checkagent

clear

# Act 1: Demo (8 tests, zero config, no API keys)
printf '$ '
sleep 0.15
for c in c h e c k a g e n t ' ' d e m o; do
    printf '%s' "$c"
    sleep 0.03
done
echo
checkagent demo 2>&1
sleep 1

# Act 2: Safety scan (68 probes, instant results)
printf '\n$ '
sleep 0.15
cmd="checkagent scan demo_agent:my_agent"
for (( i=0; i<${#cmd}; i++ )); do
    printf '%s' "${cmd:$i:1}"
    sleep 0.03
done
echo
PYTHONPATH=/tmp checkagent scan demo_agent:my_agent 2>&1
sleep 2
