import argparse
import glob
import json
import logging
import os
from typing import Dict

import tqdm

import log
from data_utils.copy4e2e import Copy4E2EF1
from mmocr.ocr import MMOCR
from work_dirs_utils.pkl2json import convert, load_pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_type',
        choices=['aihub_finance', 'aihub_transit'],
        action='append')
    parser.add_argument('--logfile_path', default='e2e.log', type=str)
    args = parser.parse_args()
    return args


def get_model_config(model_name: str) -> Dict:
    dbnet_base = 'configs/textdet/dbnet'
    f_dbnet_confname = 'dbnet_resnet18_fpnc_20e_aihubfinance10of100'
    t_dbnet_confname = 'dbnet_resnet18_fpnc_2e_aihubtransit100of100'
    sar_base = 'configs/textrecog/sar'
    f_sar_confname = 'sar_resnet31_parallel-decoder_500e_aihubfinance1of100_pretrained'  # noqa E501
    t_sar_confname = 'sar_resnet31_parallel-decoder_100e_aihubtransit1of100_pretrained'  # noqa E501
    satrn_base = 'configs/textrecog/satrn'
    f_satrn_confname = 'satrn_shallow-small_150e_aihubfinance1of100_pretrained'
    t_satrn_confname = 'satrn_shallow-small_30e_aihubtransit1of100_pretrained'
    model_dict = {
        # Detection models
        'AihubFinance_DBNet': {
            'det_config': f'{dbnet_base}/{f_dbnet_confname}.py',
            'det_ckpt':
            f'pretrained/{f_dbnet_confname}_sparkling-cloud-104.pth',
        },
        'AihubTransit_DBNet': {
            'det_config': f'{dbnet_base}/{t_dbnet_confname}.py',
            'det_ckpt': f'pretrained/{t_dbnet_confname}_hopeful-leaf-117.pth',
        },
        # Recognition models
        'AihubFinance_SAR': {
            'recog_config': f'{sar_base}/{f_sar_confname}.py',
            'recog_ckpt': f'pretrained/{f_sar_confname}_zesty-sun-97.pth',
        },
        'AihubTransit_SAR': {
            'recog_config': f'{sar_base}/{t_sar_confname}.py',
            'recog_ckpt': f'pretrained/{t_sar_confname}_stilted-grass-112.pth',
        },
        'AihubFinance_SATRN_sm': {
            'recog_config': f'{satrn_base}/{f_satrn_confname}.py',
            'recog_ckpt':
            f'pretrained/{f_satrn_confname}_proud-resonance-129.pth',
        },
        'AihubTransit_SATRN_sm': {
            'recog_config': f'{satrn_base}/{t_satrn_confname}.py',
            'recog_ckpt':
            f'pretrained/{t_satrn_confname}_driven-planet-130.pth',
        },
    }
    if model_name not in model_dict:
        raise ValueError(f'Model {model_name} is not supported.')
    else:
        return model_dict[model_name]


def get_fname(path):
    basename = os.path.basename(path)
    fname = os.path.splitext(basename)[0]
    return fname


def write_empty_json(path):

    with open(path, 'w', encoding='UTF-8-sig') as f:
        d = {
            'rec_texts': [],
            'rec_scores': [],
            'det_polygons': [],
        }
        json.dump(d, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    log.set_default_logger(level='INFO', logfile_path=args.logfile_path)
    logger = logging.getLogger('ocr_e2e')

    # Finance
    if 'aihub_finance' in args.data_type:
        logger.info('\n=== Aihub 금융 모델 로드 ===')
        kwargs = get_model_config('AihubFinance_DBNet')
        kwargs.update(get_model_config('AihubFinance_SATRN_sm'))
        ocr = MMOCR(**kwargs)
        logger.info('\n=== Aihub 금융 모델 추론 시작 ===')
        finance = Copy4E2EF1('aihub_finance')
        imgs = glob.glob(os.path.join(finance.dst_raw_img, '*.png'))
        for i, p in enumerate(tqdm.tqdm(imgs)):
            pkl_path = f'work_dirs/aihub_finance/{get_fname(p)}.pkl'
            try:
                ocr.readtext(img=p, pred_out_file=pkl_path)
            except IndexError:
                logger.warning(f'추론값이 비어 있는 이미지입니다: {p}')
                write_empty_json(pkl_path.replace('.pkl', '.json'))
            else:
                convert(load_pkl(pkl_path), pkl_path.replace('.pkl', '.json'))

    # Transit
    if 'aihub_transit' in args.data_type:
        logger.info('\n=== Aihub 물류 모델 로드 ===')
        kwargs = get_model_config('AihubTransit_DBNet')
        kwargs.update(get_model_config('AihubTransit_SATRN_sm'))
        ocr = MMOCR(**kwargs)
        logger.info('\n=== Aihub 물류 모델 추론 시작 ===')
        transit = Copy4E2EF1('aihub_transit')
        imgs = glob.glob(os.path.join(transit.dst_raw_img, '*.png'))
        for i, p in enumerate(tqdm.tqdm(imgs)):
            pkl_path = f'work_dirs/aihub_transit/{get_fname(p)}.pkl'
            try:
                ocr.readtext(img=p, pred_out_file=pkl_path)
            except IndexError:
                logger.warning(f'추론값이 비어 있는 이미지입니다: {p}')
                write_empty_json(pkl_path.replace('.pkl', '.json'))
            else:
                convert(load_pkl(pkl_path), pkl_path.replace('.pkl', '.json'))


if __name__ == '__main__':
    main()
