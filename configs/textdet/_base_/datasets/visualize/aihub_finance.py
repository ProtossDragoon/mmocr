_base_ = [
    '../../datasets/aihub_finance.py',
    '../../default_runtime.py',
]

aihubfinance_file_client_args = dict(backend='disk')

aihubfinance_visualization_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=aihubfinance_file_client_args,
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

aihubfinance_visualization_datalist = [
    # _base_.aihubfinance_textdet_sampled4vis1,
    _base_.aihubfinance_textdet_sampled4vis2,
    _base_.aihubfinance_textdet_sampled4vis3,
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=aihubfinance_visualization_datalist,
        pipeline=aihubfinance_visualization_pipeline))
