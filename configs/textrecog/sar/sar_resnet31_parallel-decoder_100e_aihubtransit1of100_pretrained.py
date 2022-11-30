_base_ = [
    '_base_sar_resnet31_parallel-decoder_aihubtransit.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/aihub_transit.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
]

load_from = ('./pretrained/'
             'sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real'
             '_20220915_171910-04eb4e75.pth')

# dataset settings
train_list = [_base_.aihubtransit1of100_textrecog_train]
val_list = [
    _base_.aihubtransit10of100_textrecog_test, _base_.icdar2015_textrecog_test
]
test_list = [
    _base_.aihubtransit1of100_textrecog_test,
    _base_.aihubtransit10of100_textrecog_test,
    _base_.aihubtransit_textrecog_sampled4vis2,
    _base_.aihubtransit_textrecog_sampled4vis3, _base_.icdar2015_textrecog_test
]

train_dataloader = dict(
    batch_size=160,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

auto_scale_lr = dict(base_batch_size=160)

val_dataloader = dict(
    batch_size=160,
    num_workers=12,
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
    dataset_prefixes=[
        'AihubTransit1of100', 'AihubTransit10of100',
        'AihubTransit100of100Sampled2', 'AihubTransit100of100Sampled3', 'IC15'
    ])

# Save checkpoints every 20 epochs
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=20))

# Set the maximum number of epochs to 100,
# and validate the model every 1 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
