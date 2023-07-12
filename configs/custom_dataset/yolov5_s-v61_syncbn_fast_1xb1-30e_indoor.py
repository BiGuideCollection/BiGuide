_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

max_epochs = 50  # maximum epochs for training
data_root = './data/user_study_indoor_all_coco/'  # absolute path to the dataset directory
# data_root = '/root/workspace/mmyolo/data/cat/'  # absolute path to the dataset dir inside the Docker container
# the path of result save, can be omitted, omitted save file name is located under work_dirs with the same name of config file.
# If a config variable changes only part of its parameters, changing this variable will save the new training file elsewhere
work_dir = './work_dirs/indoor_biguide_user1_5'

# load_from can specify a local path or URL, setting the URL will automatically download, because the above has been downloaded, we set the local path here
# since this tutorial is fine-tuning on the cat dataset, we need to use `load_from` to load the pre-trained model from MMYOLO. This allows for faster convergence and accuracy
load_from = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# according to your GPU situation, modify the batch size, and YOLOv5-s defaults to 8 cards x 16bs
train_batch_size_per_gpu = 1
train_num_workers = 1  # recommend to use train_num_workers = nGPU x 4

save_epoch_intervals = 1  # save weights every interval round

# according to your GPU situation, modify the base_lr, modification ratio is base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4

anchors = [  # the anchor has been updated according to the characteristics of dataset. The generation of anchor will be explained in the following section.
    [(83, 89), (160, 182), (234, 234)],  # P3/8
    [(288, 262), (323, 401), (392, 417)],  # P4/16
    [(456, 524), (457, 609), (544, 781)]  # P5/32
]

class_name = ('mobile_phone', 'scissors', 'light_bulb', 'tin_can', 'ball', 'mug', 'remote_control',)  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=(220, 20, 60) # the color of drawing, free to set
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=1,  # number of epochs to start validation.  Here 20 is set because the accuracy of the first 20 epochs is not high and the test is not meaningful, so it is skipped
    val_interval=save_epoch_intervals  # the test evaluation is performed  iteratively every val_interval round
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls is dynamically adjusted based on num_classes, but when num_classes = 1, loss_cls is always 0
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(  
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        # if the dataset is too small, you can use RepeatDataset, which repeats the current dataset n times per epoch, where 5 is set.
        times=5,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train_biguide_indoor_user1_5.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test_precollected_indoor.json',
        data_prefix=dict(img='images/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test_precollected_indoor.json',
        data_prefix=dict(img='images/')))

val_evaluator = dict(ann_file=data_root + 'annotations/test_precollected_indoor.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test_precollected_indoor.json')

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    # set how many epochs to save the model, and the maximum number of models to save,`save_best` is also the best model (recommended).
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=5,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    # logger output interval
    logger=dict(type='LoggerHook', interval=10))

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])

