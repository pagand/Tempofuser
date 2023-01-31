#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import json

dataset_dir = "./dataset/"
HORIZEN = 10

def _load_json(path):
        try:
            json_value = json.load(open(path))
        except Exception as e:
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(new_path))
        return json_value

def _load_json_gps(path):
        try:
            json_value = json.load(open(path))
        except Exception as e:
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(new_path))
        return [json_value["gps_x"], json_value["gps_y"]]

def fetch_future_waypoints(path, future_waypoints_cached):
    # if the catched value not available
    indx = int(path[-9:-5])
    path = path[:-9]
    if not future_waypoints_cached.all():
        for i in range(1, HORIZEN):
            future_waypoints_cached[i-1][:] = _load_json_gps(os.path.join(path,"%04d.json" % (indx+i)))
    
    return np.vstack([future_waypoints_cached,_load_json_gps(os.path.join(path,"%04d.json" % (indx+HORIZEN)))])
     





dirs = os.listdir(dataset_dir)
for dir in reversed(dirs): 
    if dir[:7] != "weather":
        dirs.remove(dir)

dirs = ["weather-0"]
for weather in dirs:
    route_dir = os.path.join(dataset_dir,weather,'data/')
    if not os.path.exists(route_dir):
        continue
    r_dirs = sorted(os.listdir(route_dir))
    for r_dir in r_dirs:

        mes_dir = os.path.join(route_dir, r_dir,'measurements/')
        mes_list = os.listdir(mes_dir)
        future_waypoints_cached = np.zeros((HORIZEN-1,2))
        for mes in mes_list:
            path = os.path.join(mes_dir,mes)
            safe_waypoints = fetch_future_waypoints(path,future_waypoints_cached)
            dic = _load_json(path)
            dic["safe_waypoints"] = safe_waypoints.tolist()
            file_handler = open(path, "w")
            json.dump(dic, file_handler)
            file_handler.close()

