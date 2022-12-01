_base_ = [
    '_base_satrn_shallow_aihubtransit.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/aihub_transit.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
]

load_from = 'pretrained/satrn_academic_20211009-cb8b1580.pth'

# dataset settings
train_list = [
    _base_.aihubtransit1of100_textrecog_train,
    _base_.aihubtransit1of100_textrecog_test,  # 멀쩡한 데이터 영끌
    _base_.aihubtransit10of100_textrecog_test,  # 멀쩡한 데이터 영끌
]
val_list = [
    _base_.aihubtransit1of100_textrecog_test, _base_.icdar2015_textrecog_test
]
test_list = [
    _base_.aihubtransit1of100_textrecog_test, _base_.icdar2015_textrecog_test
]

train_dataloader = dict(
    batch_size=64,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

auto_scale_lr = dict(base_batch_size=160)

val_dataloader = dict(
    batch_size=64,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=val_list,
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    type='MultiDatasetsEvaluator', dataset_prefixes=['AihubTransit', 'IC15'])

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_evaluator = dict(
    type='MultiDatasetsEvaluator',
    dataset_prefixes=['AihubTransit1of100', 'IC15'])

# Save checkpoints every 1 epochs
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1))

# Set the maximum number of epochs to 5, and validate the model every 1 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)

# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[1, 2, 3, 4], gamma=0.5),
]
