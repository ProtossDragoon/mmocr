_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args,
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='ImgAugWrapper', args=[['Resize', [0.8, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
