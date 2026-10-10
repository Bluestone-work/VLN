#!/usr/bin/env python
"""Small interactive Habitat-Sim viewer for inspecting project scenes.

This is deliberately separate from ETPNav evaluation. It creates one
windowed simulator, displays the RGB sensor, and accepts keyboard motion.
Keys: W/Up forward, A/Left turn left, D/Right turn right, R reset, Esc/Q quit.
"""
import argparse
import tkinter as tk
from pathlib import Path

import habitat_sim
from habitat_sim.agent import AgentConfiguration
from habitat_sim.sensor import SensorSpec, SensorType
from PIL import Image, ImageTk


def build_sim(scene, width, height):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.gpu_device_id = 0
    agent_cfg = AgentConfiguration()
    sensor = SensorSpec()
    sensor.uuid = "color_sensor"
    sensor.sensor_type = SensorType.COLOR
    sensor.resolution = [height, width]
    sensor.position = [0.0, 1.5, 0.0]
    agent_cfg.sensor_specifications = [sensor]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


class Viewer:
    def __init__(self, sim, width, height):
        self.sim, self.width, self.height = sim, width, height
        self.root = tk.Tk()
        self.root.title("ETPNav Habitat Viewer")
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.status = tk.Label(self.root, anchor="w")
        self.status.pack(fill="x")
        self.root.bind("<KeyPress>", self.key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update()

    def update(self):
        obs = self.sim.get_sensor_observations()["color_sensor"]
        image = ImageTk.PhotoImage(Image.fromarray(obs[:, :, :3]))
        self.label.configure(image=image)
        self.label.image = image
        state = self.sim.get_agent(0).get_state()
        self.status.configure(text="W/Up forward   A/Left turn   D/Right turn   R reset   Esc/Q quit   |   position: {}".format([round(float(x), 2) for x in state.position]))

    def key(self, event):
        key = event.keysym.lower()
        if key in ("escape", "q"):
            self.close(); return
        if key in ("w", "up"):
            self.sim.step("move_forward")
        elif key in ("a", "left"):
            self.sim.step("turn_left")
        elif key in ("d", "right"):
            self.sim.step("turn_right")
        elif key == "r":
            self.sim.reset()
        self.update()

    def close(self):
        self.sim.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="data/scene_datasets/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args()
    scene = Path(args.scene)
    if not scene.exists():
        raise FileNotFoundError(scene)
    Viewer(build_sim(scene, args.width, args.height), args.width, args.height).run()


if __name__ == "__main__":
    main()
