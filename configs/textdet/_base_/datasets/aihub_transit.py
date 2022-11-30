aihubtransit_textdet_data_root = 'data/det/aihub_transit'

aihubtransit1of100_textdet_train = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_1of100',
    ann_file='textdet_train.json',
    data_prefix=dict(img_path='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

aihubtransit1of100_textdet_test = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_1of100',
    ann_file='textdet_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)

aihubtransit10of100_textdet_train = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_10of100',
    ann_file='textdet_train.json',
    data_prefix=dict(img_path='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

aihubtransit10of100_textdet_test = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_10of100',
    ann_file='textdet_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)

aihubtransit100of100_textdet_train = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_100of100',
    ann_file='textdet_train.json',
    data_prefix=dict(img_path='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

aihubtransit100of100_textdet_validation = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_100of100',
    ann_file='textdet_validation.json',
    data_prefix=dict(img_path='validation/'),
    test_mode=True,
    pipeline=None)

aihubtransit100of100_textdet_test = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/part_100of100',
    ann_file='textdet_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)

sampled4vis = 'part_100of100_sampled_for_vis'

aihubtransit_textdet_sampled4vis1 = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/{sampled4vis}',
    ann_file='textdet_train.json',
    data_prefix=dict(img_path='train/'),
    test_mode=True,
    pipeline=None)

aihubtransit_textdet_sampled4vis2 = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/{sampled4vis}',
    ann_file='textdet_validation.json',
    data_prefix=dict(img_path='validation/'),
    test_mode=True,
    pipeline=None)

aihubtransit_textdet_sampled4vis3 = dict(
    type='OCRDataset',
    data_root=f'{aihubtransit_textdet_data_root}/{sampled4vis}',
    ann_file='textdet_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)
