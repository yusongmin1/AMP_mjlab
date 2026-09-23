"""Probe gamepad axes/buttons: move sticks and press buttons to see indices.

Usage: python scripts/pad_probe.py
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time

import pygame


def main() -> None:
  pygame.display.init()
  pygame.joystick.init()
  if pygame.joystick.get_count() == 0:
    print("No gamepad detected.")
    return
  j = pygame.joystick.Joystick(0)
  j.init()
  print(f"{j.get_name()}: {j.get_numaxes()} axes, {j.get_numbuttons()} buttons, {j.get_numhats()} hats")
  print("Move sticks / press buttons (Ctrl+C to quit)...\n")
  last_axes = None
  last_buttons = None
  try:
    while True:
      pygame.event.get()
      axes = tuple(round(j.get_axis(i), 2) for i in range(j.get_numaxes()))
      buttons = tuple(i for i in range(j.get_numbuttons()) if j.get_button(i))
      hats = tuple(j.get_hat(h) for h in range(j.get_numhats()))
      if axes != last_axes or buttons != last_buttons or hats != (last_hats if (last_hats := hats) else hats):
        print(f"axes={axes} buttons={buttons} hats={hats}")
        last_axes, last_buttons = axes, buttons
      time.sleep(0.03)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
