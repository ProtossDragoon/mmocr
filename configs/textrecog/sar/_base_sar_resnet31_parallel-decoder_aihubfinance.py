_base_ = [
    '_base_sar_resnet31_parallel-decoder.py',
]

_base_.dictionary = dict(
    type='Dictionary',
    dict_file=('{{ fileDirname }}/../../../dicts/'
               'korean_english_digits_symbols_space.txt'),
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True)

_base_.model.decoder.dictionary = _base_.dictionary

_base_.train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args,
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

_base_.test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
