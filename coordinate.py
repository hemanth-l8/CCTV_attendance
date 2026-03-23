import pyautogui
import time

print("Move mouse to TOP-LEFT of emulator in 5 seconds...")
time.sleep(5)
print("Top-left:", pyautogui.position())

print("Move mouse to BOTTOM-RIGHT of emulator in 5 seconds...")
time.sleep(5)
print("Bottom-right:", pyautogui.position())