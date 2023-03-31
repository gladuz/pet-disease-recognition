# Inherit and overwrite part of the config based on this config
_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = '../data/' # dataset root
class_name = ('A1', 'A2', 'A3', 'A4', 'A5', 'A6') # dataset category name
num_classes = len(class_name) # dataset category number
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_name)

base_lr = 1e-3
# Max training 40 epoch
max_epochs = 40
# bs = 12
train_batch_size_per_gpu = 32
# dataloader num workers
train_num_workers = 8

# load COCO pre-trained weight
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

model = dict(
    # Fixed the weight of the entire backbone without training
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
    ),
    train_cfg=dict(assigner=dict(num_classes=num_classes))
    )
    

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # Dataset annotation file of json path
        ann_file='train_annotation_coco.json',
        # Dataset prefix
        data_prefix=dict(img='')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid_annotation_coco.json',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'valid_annotation_coco.json')
test_evaluator = val_evaluator

default_hooks = dict(
    # Save weights every 10 epochs and a maximum of two weights can be saved.
    # The best model is saved automatically during model evaluation
    checkpoint=dict(interval=2, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    # The log printing interval is 5
    logger=dict(type='LoggerHook', interval=5))
# The evaluation interval is 10
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])