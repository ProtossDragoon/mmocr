_base_ = [
    '_base_satrn_shallow-small.py',
]

_base_.dictionary = dict(
    type='Dictionary',
    dict_file=('{{ fileDirname }}/../../../dicts/'
               'english_digits_symbols.txt'),
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
        height=20,
        min_width=10,
        max_width=180,
        width_divisor=4),
    dict(type='PadToWidth', width=180),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

_base_.test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='RescaleToHeight',
        height=20,
        min_width=10,
        max_width=180,
        width_divisor=4),
    dict(type='PadToWidth', width=180),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
