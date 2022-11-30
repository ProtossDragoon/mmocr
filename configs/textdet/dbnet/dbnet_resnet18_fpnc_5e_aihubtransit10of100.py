_base_ = [
    '_base_dbnet_resnet18_fpnc_aihubtransit.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/aihub_transit.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

load_from = None

# dataset settings
train_list = [_base_.aihubtransit10of100_textdet_train]
val_list = [
    _base_.aihubtransit10of100_textdet_test, _base_.icdar2015_textdet_test
]
test_list = [
    _base_.aihubtransit_textdet_sampled4vis3, _base_.icdar2015_textdet_test
]

train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

auto_scale_lr = dict(base_batch_size=32)

val_dataloader = dict(
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset', datasets=val_list,
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=dict(type='HmeanIOUMetric'),
    dataset_prefixes=['AihubTransit', 'IC15'])

test_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=dict(type='HmeanIOUMetric'),
    dataset_prefixes=['AihubTransit100of100RandomSampled', 'IC15'])

# Save checkpoints every 1 epochs
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=1))

# Set the maximum number of epochs to 5, and validate the model every 1 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
