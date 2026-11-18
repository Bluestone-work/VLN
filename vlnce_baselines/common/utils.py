from typing import Any, Dict, List
import torch
import torch.distributed as dist
import numpy as np
import copy
import math

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    max_length: int = 512,
    pad_id: int = 0,
    task_type: int = None,
):
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure."""
    if instruction_sensor_uuid not in observations[0]:
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
            origin_token_len = len(token)
            if len(token) < max_length:
                token += [pad_id] * (max_length - len(token))
            observations[i][instruction_sensor_uuid] = token
            if task_type in (1, 2):
                task_encoding = [task_type] * origin_token_len
                task_encoding += [0] * (max_length - len(task_encoding))
                observations[i]["txt_task_encoding"] = task_encoding
        else:
            break
    return observations

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor)
    return gather_t

def dis_to_con(path, amount=0.25):
    starts = path[:-1]
    ends = path[1:]
    new_path = [path[0]]
    for s, e in zip(starts,ends):
        vec = np.array(e) - np.array(s)
        ratio = amount/np.linalg.norm(vec[[0,2]])
        unit = vec*ratio
        times = int(1/ratio)
        for i in range(times):
            if i != times - 1:
                location = np.array(new_path[-1])+unit
                new_path.append(location.tolist())
        new_path.append(e)
    
    return new_path

def get_camera_orientations12():
    base_angle_deg = 30
    base_angle_rad = math.pi / 6
    orient_dict = {}
    for k in range(1,12):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict
