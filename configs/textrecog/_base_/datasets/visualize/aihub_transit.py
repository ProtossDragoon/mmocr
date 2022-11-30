_base_ = [
    '../../datasets/aihub_transit.py',
    '../../default_runtime.py',
]

aihubtransit_file_client_args = dict(backend='disk')

aihubtransit_visualization_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=aihubtransit_file_client_args,
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

aihubtransit_visualization_datalist = [
    # _base_.aihubtransit_textrecog_sampled4vis1,
    _base_.aihubtransit_textrecog_sampled4vis2,
    _base_.aihubtransit_textrecog_sampled4vis3,
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=aihubtransit_visualization_datalist,
        pipeline=aihubtransit_visualization_pipeline))
