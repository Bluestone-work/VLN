#!/usr/bin/env python
"""Small interactive Habitat-Sim viewer for inspecting project scenes.

This is deliberately separate from ETPNav evaluation. It creates one
windowed simulator, displays the RGB sensor, and accepts keyboard motion.
Keys: W/Up forward, A/Left turn left, D/Right turn right, R reset, Esc/Q quit.
"""
import argparse
import json
import tkinter as tk
import numpy as np
import quaternion
from pathlib import Path

import habitat_sim
from habitat_sim.agent import AgentConfiguration
from habitat_sim.sensor import SensorSpec, SensorType
from habitat_sim.agent import AgentState
from PIL import Image, ImageTk


def unpack(value):
    if isinstance(value, dict) and "__array__" in value:
        return value["__array__"]
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, dict):
        return {k: unpack(v) for k, v in value.items()}
    return value


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
    depth = SensorSpec()
    depth.uuid = "depth_sensor"
    depth.sensor_type = SensorType.DEPTH
    depth.resolution = [height, width]
    depth.position = [0.0, 1.5, 0.0]
    agent_cfg.sensor_specifications.append(depth)
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


class Viewer:
    def __init__(self, sim, width, height, trace=None, graph_rows=None, top_k=5):
        self.sim, self.width, self.height = sim, width, height
        self.trace = trace or []
        self.trace_index = 0
        self.playing = False
        self.graph_rows = graph_rows or []
        self.graph_index = 0
        self.top_k = top_k
        self.root = tk.Tk()
        self.root.title("ETPNav Habitat Viewer")
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        self.label = tk.Label(self.frame, bg="black")
        self.label.pack(side="left")
        self.graph = tk.Canvas(self.frame, width=420, height=600, bg="white")
        self.graph.pack(side="right")
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
        self.draw_graph()
        state = self.sim.get_agent(0).get_state()
        mode = "ETPNav replay: Space step, P play" if self.trace else "manual: W/Up forward, A/Left, D/Right"
        finished = bool(self.graph_rows and self.graph_index >= len(self.graph_rows))
        shown_decision = min(self.graph_index, len(self.graph_rows))
        suffix = "   |   trajectory finished" if finished else ""
        self.status.configure(text="{}   R reset   Esc/Q quit   |   decision {}/{} primitive {}/{}   position: {}{}".format(mode, shown_decision, len(self.graph_rows), self.trace_index, len(self.trace), [round(float(x), 2) for x in state.position], suffix))
        if self.playing:
            self.root.after(80, self.advance_trace)

    def advance_trace(self):
        if self.trace_index >= len(self.trace):
            self.playing = False
            return
        if self.graph_rows and self.graph_index < len(self.graph_rows):
            for primitive in self.graph_rows[self.graph_index].get("primitives", []):
                action = int(primitive["action"])
                self.sim.step({1: "move_forward", 2: "turn_left", 3: "turn_right"}.get(action, "move_forward"))
                self.trace_index += 1
            self.graph_index += 1
        else:
            action = int(self.trace[self.trace_index]["action"])
            self.sim.step({1: "move_forward", 2: "turn_left", 3: "turn_right"}.get(action, "move_forward"))
            self.trace_index += 1
        self.update()

    def key(self, event):
        key = event.keysym.lower()
        if key in ("escape", "q"):
            self.close(); return
        if key == "space" and self.trace:
            self.advance_trace(); return
        if key == "p" and self.trace:
            self.playing = not self.playing
            if self.playing: self.advance_trace()
            return
        if key in ("w", "up"):
            self.sim.step("move_forward")
        elif key in ("a", "left"):
            self.sim.step("turn_left")
        elif key in ("d", "right"):
            self.sim.step("turn_right")
        elif key == "r":
            self.sim.reset()
            self.trace_index = 0
            self.graph_index = 0
            self.playing = False
            if self.trace:
                first = self.trace[0]["pre_pose"]
                state = AgentState()
                state.position = np.asarray(first["position"], dtype=np.float32)
                q = first["rotation"]
                state.rotation = quaternion.quaternion(q[3], q[0], q[1], q[2])
                self.sim.get_agent(0).set_state(state)
        self.update()

    def draw_graph(self):
        self.graph.delete("all")
        if not self.graph_rows:
            self.graph.create_text(210, 25, text="No graph trace loaded")
            return
        row = self.graph_rows[min(self.graph_index, len(self.graph_rows) - 1)]
        points = []
        admissible = [o for o in row.get("options", []) if o.get("admissible", True)]
        selected_index = int(row.get("effective_index", -99))
        ranked = sorted(admissible, key=lambda o: (-float(o.get("logit", float("-inf"))), int(o.get("index", -1))))
        visible_options = {}
        for option in ranked[:self.top_k]:
            action = unpack(option.get("action", {}))
            pos = action.get("ghost_pos", action.get("stop_pos"))
            if pos is not None and hasattr(self.sim, "pathfinder"):
                try:
                    if not self.sim.pathfinder.is_navigable(np.asarray(pos, dtype=np.float32)):
                        continue
                except Exception:
                    pass
            visible_options[int(option.get("index", -1))] = option
        for option in admissible:
            if int(option.get("index", -1)) == selected_index:
                visible_options[selected_index] = option
        for option in visible_options.values():
            action = unpack(option.get("action", {})); pos = action.get("ghost_pos", action.get("stop_pos"))
            if pos is not None: points.append((float(pos[0]), float(pos[2])))
        gp = unpack(row.get("graph_position"))
        if isinstance(gp, dict): gp = gp.get("position")
        if gp is None: gp = self.sim.get_agent(0).get_state().position
        points.append((float(gp[0]), float(gp[2])))
        obstacle_points = []
        depth = self.sim.get_sensor_observations().get("depth_sensor")
        if depth is not None:
            h, w = depth.shape[:2]
            rotation = self.sim.get_agent(0).get_state().rotation
            for v in range(8, h - 4, 12):
                for u in range(4, w - 4, 12):
                    d = float(depth[v, u])
                    if not (0.15 < d < 3.0): continue
                    lateral = (u - (w - 1) / 2.0) / max(w, 1) * d * 1.6
                    world_ray = quaternion.rotate_vectors(rotation, np.asarray([lateral, 0.0, -d]))
                    obstacle_points.append((float(gp[0] + world_ray[0]), float(gp[2] + world_ray[2])))
        points.extend(obstacle_points)
        if not points: return
        xs, ys = zip(*points); xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys); scale = min(360/max(xmax-xmin,1e-3), 520/max(ymax-ymin,1e-3))
        def xy(p): return (30+(p[0]-xmin)*scale, 570-(p[1]-ymin)*scale)
        self.graph.create_text(210, 15, text="ETPNav graph decision {}  selected index {}".format(row.get("high_level_step"), row.get("effective_index")), fill="black")
        x,y=xy((gp[0],gp[2])); self.graph.create_oval(x-7,y-7,x+7,y+7,fill="blue"); self.graph.create_text(x+10,y,text="current",anchor="w")
        for option in visible_options.values():
            action=unpack(option.get("action",{})); pos=action.get("ghost_pos",action.get("stop_pos"))
            if pos is None: continue
            x,y=xy((float(pos[0]),float(pos[2]))); idx=int(option.get("index",-1)); selected=idx==int(row.get("effective_index",-99)); color="red" if selected else ("orange" if option.get("current_proposal") else "gray")
            self.graph.create_oval(x-5,y-5,x+5,y+5,fill=color); self.graph.create_text(x+8,y,text="{}:{:.2f}".format(idx,float(option.get("logit",0))),anchor="w",fill="black")
        # Project the current depth observation into a small local top-down
        # obstacle cloud around the current node. This is observation-derived
        # geometry, not privileged map or reference-path information.
        if obstacle_points:
            for obstacle in obstacle_points:
                ox, oy = xy(obstacle)
                self.graph.create_oval(ox - 1, oy - 1, ox + 1, oy + 1, fill="#00a6a6", outline="")
            self.graph.create_text(210, 590, text="cyan: visible depth returns; top-{} admissible actions".format(self.top_k), fill="#007777")

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
    p.add_argument("--trace", type=Path, help="ETPNav primitive trace JSONL for replay")
    p.add_argument("--graph-options", type=Path, help="ETPNav graph_options.jsonl for decision visualization")
    p.add_argument("--episode-id", default="1778")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    scene = Path(args.scene)
    if not scene.exists():
        raise FileNotFoundError(scene)
    trace = []
    trace_rows = []
    if args.trace:
        for line in args.trace.open():
            row = json.loads(line)
            if str(row.get("episode_id")) == str(args.episode_id):
                trace_rows.append(row)
                trace.extend({"action": x["action"], "pre_pose": row["pre_pose"]} for x in row.get("primitives", []))
    graph_rows = []
    if args.graph_options:
        for line in args.graph_options.open():
            row = json.loads(line)
            if str(row.get("episode_id")) == str(args.episode_id):
                graph_rows.append(row)
    for row in trace_rows:
        match = next((g for g in graph_rows if int(g.get("high_level_step", -1)) == int(row.get("high_level_step", -2))), None)
        if match is not None: match["primitives"] = row.get("primitives", [])
    viewer = Viewer(build_sim(scene, args.width, args.height), args.width, args.height, trace, graph_rows, args.top_k)
    if trace:
        state = AgentState()
        state.position = np.asarray(trace[0]["pre_pose"]["position"], dtype=np.float32)
        q = trace[0]["pre_pose"]["rotation"]
        state.rotation = quaternion.quaternion(q[3], q[0], q[1], q[2])
        viewer.sim.get_agent(0).set_state(state)
        viewer.update()
    viewer.run()


if __name__ == "__main__":
    main()
