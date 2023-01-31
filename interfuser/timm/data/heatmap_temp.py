import math
import json
import os

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

import copy

from skimage.measure import block_reduce


# VALUES = [255, 150, 120, 90, 60, 30][::-1]
# EXTENT = [0, 0.2, 0.4, 0.6, 0.8, 1.0][::-1]


VALUES = [255]
EXTENT = [0]


def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    img_size = max_distance * pixels_per_meter * 2
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = np.around(left_up).astype(np.int)
    left_down = np.around(left_down).astype(np.int)
    right_down = np.around(right_down).astype(np.int)
    right_up = np.around(right_up).astype(np.int)
    left_up = list(left_up)
    left_down = list(left_down)
    right_up = list(right_up)
    right_down = list(right_down)
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img


def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw



def generate_hmap(measurements, actors_data, pixels_per_meter=5, max_distance=18):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.int)
    ego_x = measurements["x"]
    ego_y = measurements["y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    ego_id = None
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  
        color = np.array([0, 0, 0]) # if actor not defined, it's zero

        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 1:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        if int(_id) in measurements["is_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_bike_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_junction_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_pedestrian_present"]:
            color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * 1.5  # FIXME enlarge the size of pedstrian and bike
        # enlarge the size of the pedestrain and bikes
        if int(_id) in measurements["is_bike_present"] or measurements["is_pedestrian_present"]:
            box = box * 1.5

        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(
                act_img,
                loc,
                ori,
                box + EXTENT[i],
                VALUES[i],
                pixels_per_meter,
                max_distance,
                color,
            )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img[:, :, 0]
    return img




def convert_grid_to_xy(i, j):
    x = j - 9.5
    y = 17.5 - i
    return x, y




def generate_thmap_data(
    measurements, actors_data, pixels_per_meter=5, max_distance=18
): 
    heatmap = generate_hmap(copy.deepcopy(measurements), copy.deepcopy(actors_data))

    traffic_heatmap = block_reduce(heatmap, block_size=(5, 5), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:20, 8:28]
    # det_data = np.zeros((20, 20, 7))  
    thmap = np.zeros((20, 20, 1)) # TODO add more for the prediction

    ego_x = measurements["x"]
    ego_y = measurements["y"]
    ego_theta = measurements["theta"]
    ego_speed = measurements["speed"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    need_deleted_ids = []
    for _id in actors_data:
            
        raw_loc = actors_data[_id]["loc"]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        dis = new_loc[0] ** 2 + new_loc[1] ** 2
        if (
            dis <= 1
            or dis >= (max_distance + 3) ** 2 * 2
            or "box" not in actors_data[_id]
            or actors_data[_id]["tpe"] == 2  # no light actor
        ):
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])


    for _id in need_deleted_ids:
        del actors_data[_id]

    thmap = copy.deepcopy(traffic_heatmap)
    # if the i,j are insidet the box of the min_id >> put 1
    thmap[thmap!=0] = 1



    for i in range(20):  # Vertical
        for j in range(20):  # horizontal
            # if traffic_heatmap[i][j] < 0.05 * 255.0:
            #     continue
            if thmap[i][j]:
                continue
            center_x, center_y = convert_grid_to_xy(i, j)
            min_dis = 1000
            min_id = None
            for _id in actors_data:
                loc = actors_data[_id]["loc"][:2]
                ori = actors_data[_id]["ori"][:2]
                box = actors_data[_id]["box"]
                dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = _id
            
            # cut-off for large distance
            if min_dis> max_distance:
                continue
            
            # if the i,j are out the box but in the direction of the velocity put probability based on distance
            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            speed = np.linalg.norm(actors_data[min_id]["vel"])

            rel_speed_y = speed* np.cos(theta) - ego_speed*np.cos(ego_theta) # horizental
            rel_speed_x = speed* np.sin(theta) - ego_speed*np.sin(ego_theta) # vertical

            rel_speed = (rel_speed_x **2 + rel_speed_y **2)*10**12
            rel_theta = np.arctan2(rel_speed_y,rel_speed_x)

            theta_ij =  np.arctan2(-loc[1] + center_y , -loc[0] + center_x)
            d = np.abs(rel_theta - theta_ij)
            if  d > np.pi/2:
                d = np.abs(rel_theta - theta_ij + 2*np.pi)
                if d > np.pi/2 :
                    d = np.abs(rel_theta - theta_ij - 2*np.pi) 
                    if d > np.pi/2:
                        d = np.abs(rel_theta - theta_ij)
            # so it is within the pi rage of heading
            effects_theta = np.power(0.5 / max(0.5, np.sqrt(d)), 0.5)
            effects_dist = np.power(0.5*rel_speed / max(0.5*rel_speed, np.sqrt(min_dis)), 0.5)
            effects_total = effects_theta*effects_dist
            thmap[i][j] = np.array([effects_total])

    return thmap
