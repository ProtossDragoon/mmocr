aihubfinance_textrecog_data_root = 'data/rec/aihub_finance'

aihubfinance1of100_textrecog_train = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_1of100',
    ann_file='textrecog_train.json',
    data_prefix=dict(img_path='train/'),
    pipeline=None)

aihubfinance1of100_textrecog_test = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_1of100',
    ann_file='textrecog_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)

aihubfinance10of100_textrecog_train = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_10of100',
    ann_file='textrecog_train.json',
    data_prefix=dict(img_path='train/'),
    pipeline=None)

aihubfinance10of100_textrecog_test = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_10of100',
    ann_file='textrecog_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)

aihubfinance100of100_textrecog_train = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_100of100',
    ann_file='textrecog_train.json',
    data_prefix=dict(img_path='train/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

aihubfinance100of100_textrecog_validation = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_100of100',
    ann_file='textrecog_validation.json',
    data_prefix=dict(img_path='validation/'),
    test_mode=True,
    pipeline=None)

aihubfinance100of100_textrecog_test = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/part_100of100',
    ann_file='textrecog_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)

sampled4vis = 'part_100of100_sampled_for_vis'

aihubfinance_textrecog_sampled4vis1 = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/{sampled4vis}',
    ann_file='textrecog_train.json',
    data_prefix=dict(img_path='train/'),
    test_mode=True,
    pipeline=None)

aihubfinance_textrecog_sampled4vis2 = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/{sampled4vis}',
    ann_file='textrecog_validation.json',
    data_prefix=dict(img_path='validation/'),
    test_mode=True,
    pipeline=None)

aihubfinance_textrecog_sampled4vis3 = dict(
    type='OCRDataset',
    data_root=f'{aihubfinance_textrecog_data_root}/{sampled4vis}',
    ann_file='textrecog_test.json',
    data_prefix=dict(img_path='test/'),
    test_mode=True,
    pipeline=None)
