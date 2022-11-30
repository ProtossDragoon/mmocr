_base_ = [
    '../../datasets/aihub_finance.py',
    '../../default_runtime.py',
]

aihubfinance_file_client_args = dict(backend='disk')

aihubfinance_visualization_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=aihubfinance_file_client_args,
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

aihubfinance_visualization_datalist = [
    # _base_.aihubfinance_textrecog_sampled4vis1,
    _base_.aihubfinance_textrecog_sampled4vis2,
    _base_.aihubfinance_textrecog_sampled4vis3,
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
