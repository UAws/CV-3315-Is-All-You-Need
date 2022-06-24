_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/kitti_seg_basic.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py',
    '../_base_/wandb_logger_mmseg_training_kitti_segFormer.py'
]

log_config = {{_base_.customized_log_config}}

# optimizer
crop_size = (864, 256)

val_interval = 500

runner = dict(type='IterBasedRunner', max_iters=20000)

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=100)
workflow = [('train', val_interval), ('val', 1)]
evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint_config = dict(_delete_=True)
data = dict(samples_per_gpu=8, workers_per_gpu=4)
