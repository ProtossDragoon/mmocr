import argparse
import glob
import json
import logging
import os

import tqdm

import log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_type',
        choices=['aihub_finance', 'aihub_transit'],
        action='append')
    parser.add_argument('--resize-json', action='store_true', default=False)
    parser.add_argument('--logfile_path', default='e2e.log', type=str)
    args = parser.parse_args()
    return args


class ValidateAihubData:

    def __init__(self, data_type: str, split_type: str) -> None:
        assert data_type in ['aihub_finance', 'aihub_transit'
                             ], (f'지원되지 않는 데이터 타입입니다. 이 클래스는 aihub 데이터를 '
                                 f'처리할 수 있도록 만들어졌습니다. 현재 값: {data_type}')
        self.data_type = data_type
        self.split_type = split_type
        self.json_root = f'data/det/{self.data_type}/part_100of100'  # noqa E501
        self.impath_root = f'data/det/{self.data_type}/part_100of100/{split_type}'  # noqa E501
        self.logger = logging.getLogger(self.__class__.__name__)

        assert os.path.isdir(self.impath_root), (
            f'이미지 경로가 존재하지 않습니다. 현재 경로: {self.impath_root}')

    def _get_impath_iterator(self, impath_key: str = 'img_path'):
        p = f'{self.json_root}/textdet_test.json'
        with open(p, 'r') as f:
            data = json.load(f)
        self.logger.info(f"json 파일(`{p}`)에서 {len(data['data_list'])}개의 "
                         '이미지에 대한 레이블이 확인되었습니다.')
        for e in tqdm.tqdm(data['data_list']):
            yield e[impath_key]

    def _get_full_path(self, impath: str) -> str:
        return os.path.join(self.impath_root, impath)

    def _validate_imexists(self, impath):
        if not os.path.isfile(impath):
            raise FileNotFoundError(f'이미지 `{impath}`가 존재하지 않습니다.')
        return impath

    def _validate_json_exists(self, json_path):
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f'json 파일 `{json_path}`가 존재하지 않습니다.')
        return json_path

    def get_impaths(self) -> list:
        li = []
        for e in self._get_impath_iterator():
            li.append(self._validate_imexists(self._get_full_path(e)))
        return li


class Copy4E2EF1(ValidateAihubData):
    """이 클래스는 추후 DataPreparer 과 통합되어야 합니다."""

    def __init__(self, data_type: str, split_type: str = 'test') -> None:
        assert split_type in ['train', 'test', 'validation'], (  # noqa E501
            f'split_type은 `train`, `test`, `validation` 중 하나여야 합니다. '
            f'현재 값: {split_type}')
        super().__init__(data_type, split_type)
        self.copy_dst_root = f'data/e2ef1/{self.data_type}/part_100of100/{self.split_type}'  # noqa E501
        self._directories = {}

    @property
    def dst_raw_img(self):
        return self._assert_exists(f'{self.copy_dst_root}/imgs')

    @property
    def dst_raw_json(self):
        return self._assert_exists(f'{self.copy_dst_root}/jsons')

    @property
    def dst_icdar_gt(self):
        return self._assert_exists(f'{self.copy_dst_root}/gts')

    @property
    def dst_icdar_pred(self):
        return self._assert_exists(f'{self.copy_dst_root}/preds')

    @property
    def dst_pred_vis(self):
        return self._assert_exists(f'{self.copy_dst_root}/vis')

    def _assert_exists(self, path):
        if not os.path.exists(path):
            self.logger.warning(f'`{path}` 경로가 존재하지 않아 새로운 디렉토리를 생성합니다.')
            os.makedirs(path, exist_ok=True)
        assert os.path.isdir(path), f'`{path}` 는 디렉토리가 아닙니다.'
        return path

    def to_dst(self, entities: list, entity='raw_img'):
        assert entity in [
            'raw_img', 'raw_json'
        ], (f'entity는 `raw_img` 또는 `raw_json` 중 하나여야 합니다. 현재 값: {entity}')
        if entity == 'raw_img':
            dst = self.dst_raw_img
        elif entity == 'raw_json':
            dst = self.dst_raw_json
        else:
            raise ValueError(f'현재 값: {entity}')
        self.logger.info(f'{len(entities)} 개의 엔티티({entity})를 `{dst}`로 복사합니다.')
        for src in tqdm.tqdm(entities):
            os.system(f'cp {src} {dst}')

    def convert_gathered_json_to_icdar_gt(self, label_scale=1.0):
        json_paths = glob.glob(f'{self.dst_raw_json}/*.json')
        self.logger.info(f'{len(json_paths)}개의 json 파일들을 발견했습니다. '
                         f'(디렉토리:`{self.dst_raw_json}`)')
        self.logger.info(f'{len(json_paths)}개의 json 파일을 icdar gt 형식으로 변환합니다.')
        if label_scale != 1.0:
            self.logger.warning(f'icdar gt 파일의 label 은 원본 json 파일 label 에서 '
                                f'{label_scale} 배 변환되어 저장됩니다.')
        for json_path in tqdm.tqdm(json_paths):
            filename = os.path.basename(json_path).replace('.json', '.txt')
            dst = os.path.join(self.dst_icdar_gt, filename)
            data = json.load(open(json_path, 'r', encoding='UTF-8-sig'))
            with open(dst, 'w', encoding='UTF-8-sig') as f:
                for bbox in data['bbox']:
                    x1, x2, x3, x4 = [int(e * label_scale) for e in bbox['x']]
                    y1, y2, y3, y4 = [int(e * label_scale) for e in bbox['y']]
                    txt = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{bbox['data']}"  # noqa E501
                    f.write(txt + '\n')

    def gather_gt(self, impaths: list, gather_from=None):
        """학습에 사용되는 json 레이블은 학습을 위해 이상한 데이터를 잘 골라내어 문제가 없지만, 원본 데이터는 오히려 손을 대지
        않아 문제가 있는 경우가 있습니다.

        이는 DVC 가 잘 되지 않는 상황을 의미하기도 합니다. 하지만 이런 상황을 극복하기 위해 DVC 를 도입할 수 없는 상황에서,
        정상적이라고 알려진 학습용 레이블에서부터 raw 데이터 더미에서 json 파일을 골라가기 위해 사용합니다.
        """
        if gather_from is None:
            gather_from = f'data/raw/{self.data_type}/part_100of100/{self.split_type}/jsons'  # noqa E501

        def _get_json_full_path(impath: str):
            basename = os.path.basename(impath)
            fname = os.path.splitext(basename)[0]
            return os.path.join(gather_from, f'{fname}.json')

        entities = []
        for p in tqdm.tqdm(impaths):
            try:
                json_path = self._validate_json_exists(_get_json_full_path(p))
            except FileNotFoundError as e:
                self.logger.error(
                    f'Hint: 원본 json 파일은 디렉토리 `{gather_from}` 으로부터 탐색됩니다.')
                raise e
            entities.append(json_path)
        assert len(impaths) == len(entities), (
            f'평가용 이미지의 갯수({len(impaths)})와 '
            f'GT json 파일의 갯수({len(entities)})가 다릅니다.')
        self.to_dst(entities, entity='raw_json')


def main():
    args = parse_args()
    log.set_default_logger(level='INFO', logfile_path=args.logfile_path)
    label_scale = 0.5 if args.resize_json else 1.0
    if 'aihub_finance' in args.data_type:
        print('\n=== Aihub 금융 테스트용 데이터 ===')
        finance_data = Copy4E2EF1(data_type='aihub_finance')
        li = finance_data.get_impaths()
        finance_data.to_dst(li, entity='raw_img')
        finance_data.gather_gt(li)
        finance_data.convert_gathered_json_to_icdar_gt(label_scale=label_scale)
    if 'aihub_transit' in args.data_type:
        print('\n=== Aihub 물류 테스트용 데이터 ===')
        transit_data = Copy4E2EF1(data_type='aihub_transit')
        li = transit_data.get_impaths()
        transit_data.to_dst(li, entity='raw_img')
        transit_data.gather_gt(li)
        transit_data.convert_gathered_json_to_icdar_gt(label_scale=label_scale)


if __name__ == '__main__':
    main()
