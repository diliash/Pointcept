class_names = [
    "drawer",
    "door",
    "lid",
    "base"
]
num_classes = 4
segment_ignore_index = (-1,)

# model settings
model = dict(
    type="PG-v1m1",
    backbone=dict(
        type="Swin3D-v1m1",
        in_channels=6,
        num_classes=4,
        base_grid_size=0.02,
        depths=[2, 4, 9, 4, 4],
        channels=[48, 96, 192, 384, 384],
        num_heads=[6, 6, 12, 24, 24],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        drop_path_rate=0.3,
        up_k=3,
        num_layers=5,
        stem_transformer=True,
        down_stride=3,
        upsample="linear_attn",
        knn_down=True,
        cRSE="XYZ_NORM",
        fp16_mode=1,
    ),
    # backbone_out_channels=96,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=50,
)


dataset_type = "PartNetSimDataset"
data_root = "data/partnetsim"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_discrete_coord=True,
                keys=("coord", "color", "normal", "segment", "instance"),
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "discrete_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_center",
                    "bbox",
                ),
                feat_keys=("coord", "normal"),
                coord_feat_keys=("coord", "normal",),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_discrete_coord=True,
                keys=("coord", "color", "normal", "instance"),
            ),
            crop=None,
            aug_transform=[
            [dict(type="CenterShift", apply_z=True)],
            ],
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "discrete_coord"),
                    feat_keys=("coord", "normal"),
                    coord_feat_keys=("coord", "normal",),
                ),
            ]
        ),
        test_mode=True,
    ),
)