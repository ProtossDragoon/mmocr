_base_ = [
    '_base_sar_resnet31_parallel-decoder_aihubfinance.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/aihub_finance.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
]

load_from = ('./pretrained/'
             'sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real'
             '_20220915_171910-04eb4e75.pth')

# dataset settings
train_list = [_base_.aihubfinance1of100_textrecog_train]
val_list = [
    _base_.aihubfinance10of100_textrecog_test, _base_.icdar2015_textrecog_test
]
test_list = [
    _base_.aihubfinance1of100_textrecog_test,
    _base_.aihubfinance10of100_textrecog_test,
    _base_.aihubfinance100of100_textrecog_test, _base_.icdar2015_textrecog_test
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

auto_scale_lr = dict(base_batch_size=128)

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset', datasets=val_list,
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    type='MultiDatasetsEvaluator', dataset_prefixes=['AihubFinance', 'IC15'])

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
        'AihubFinance1of100', 'AihubFinance10of100', 'AihubFinance100of100',
        'IC15'
    ])

# Save checkpoints every 50 epochs
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=50))

# Set the maximum number of epochs to 500,
# and validate the model every 1 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=1)
