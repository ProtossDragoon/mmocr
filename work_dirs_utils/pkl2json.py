import argparse
import json

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert pickle to json')
    parser.add_argument('pkl_path', help='path to pickle file')
    parser.add_argument('json_path', help='path to json file')
    args = parser.parse_args()
    return args


def load_pkl(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def set_default(obj):
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f'{type(obj)} is unsupported for json dump')


def convert(data: dict, json_path):
    with open(json_path, 'w', encoding='UTF-8-sig') as f:
        json.dump(data, f, default=set_default, indent=4, ensure_ascii=False)
    return json.dumps(data, default=set_default, ensure_ascii=False)


if __name__ == '__main__':
    args = parse_args()
    data = load_pkl(args.pkl_path)
    convert(data, args.json_path)
