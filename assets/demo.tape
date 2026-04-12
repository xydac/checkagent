# CheckAgent Demo Animation
# Generates assets/demo.gif

Output assets/demo.gif

Require checkagent

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 20
Set Framerate 30
Set PlaybackSpeed 1.5
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

# Act 1: Run the built-in demo
Type "checkagent demo"
Enter
Sleep 4s

# Brief pause to admire the output
Sleep 2s

# Act 2: Safety scan
Type "checkagent scan assets/demo_agent.py:my_agent"
Enter
Sleep 4s

# Let the viewer read the results
Sleep 3s
