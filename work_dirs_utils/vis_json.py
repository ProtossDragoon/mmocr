import argparse
import json

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Convert pickle to json')
    parser.add_argument('json_path', help='path to json file')
    parser.add_argument('img_path', help='path to image file')
    args = parser.parse_args()
    return args


def vis(img, label):
    plt.figure(figsize=(12, 16))
    plt.imshow(img)
    for poly in label['det_polygons']:
        (x1, y1, x2, y2, x3, y3, x4, y4) = poly
        x = (x1, x2, x3, x4)
        y = (y1, y2, y3, y4)
        plt.plot(x, y, linewidth=1, color='r', linestyle='-')
    plt.savefig('test.png')


def main(args):
    img = plt.imread(args.img_path)
    label = json.load(open(args.json_path, 'r', encoding='UTF-8-sig'))
    vis(img, label)
    print(label)


if __name__ == '__main__':
    args = parse_args()
    main(args)
