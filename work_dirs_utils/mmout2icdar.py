import argparse
import glob
import json
import logging
import os

import tqdm

import log
from data_utils.copy4e2e import Copy4E2EF1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_type',
        choices=['aihub_finance', 'aihub_transit'],
        action='append')
    parser.add_argument('--logfile_path', default='e2e.log', type=str)
    args = parser.parse_args()
    return args


def convert(dataset):
    d = Copy4E2EF1(dataset)
    for p in tqdm.tqdm(glob.glob(f'work_dirs/{dataset}/*.json')):
        data = json.load(open(p, 'r', encoding='UTF-8-sig'))
        filename = os.path.basename(p).replace('.json', '.txt')
        dst = os.path.join(d.dst_icdar_pred, filename)
        with open(dst, 'w', encoding='UTF-8-sig') as f:
            for idx, polygon in enumerate(data['det_polygons']):
                (x1, y1, x2, y2, x3, y3, x4, y4) = [int(x) for x in polygon]
                txt = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{data['rec_texts'][idx]}"  # noqa E501
                f.write(txt + '\n')


def main():
    args = parse_args()
    log.set_default_logger(level='INFO', logfile_path=args.logfile_path)
    logger = logging.getLogger('ocr_e2e')
    if 'aihub_finance' in args.data_type:
        logger.info('\n=== MMOCR 타입 Aihub 금융데이터 추론값을 ICDAR 타입으로 변경 ===')
        convert('aihub_finance')
    if 'aihub_transit' in args.data_type:
        logger.info('\n=== MMOCR 타입 Aihub 물류데이터 추론값을 ICDAR 타입으로 변경 ===')
        convert('aihub_transit')


if __name__ == '__main__':
    main()
