# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import argparse

from .motip import MOTIP
from structures.args import Args


from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder


torch.serialization.add_safe_globals([argparse.Namespace])

def build(config: dict):
    # Generate DETR args:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]

    detr_framework = config["DETR_FRAMEWORK"].lower()

    #UPDATE 2: import the same config from the previous finetuning of rf-detr small 
    match detr_framework:
        case "deformable_detr":
            from models.deformable_detr.deformable_detr import build as build_deformable_detr
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
    #UPDATE 2: import the same config from the previous finetuning of rf-detr small 
    #UPDATE 2.1: add the rf_detr case and associate the config
        case "rf_detr":
            from models.rfdetr.models.lwdetr import build_model, build_criterion_and_postprocessors
            ckpt_path = config["CKPT_PATH"]
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            args_ckpt = ckpt["args"]
            args_ckpt.num_classes -= 1
            detr = build_model(args=args_ckpt)
            args_ckpt.num_classes += 1
            detr_criterion, _ = build_criterion_and_postprocessors(args=args_ckpt)
            ckpt_model = ckpt.get("model", None)
            if ckpt_model is not None:
                # Filter out class-head keys that mismatch num_classes between checkpoint and model
                model_state = detr.state_dict()
                filtered = {}
                for k, v in ckpt_model.items():
                    bare_k = k[5:] if k.startswith('detr.') else k
                    if bare_k in model_state and v.shape == model_state[bare_k].shape:
                        filtered[bare_k] = v
                missing, unexpected = detr.load_state_dict(filtered, strict=False)
                print(f"[build] RF-DETR weights loaded. Matched: {len(filtered)}, "
                    f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            else:
                print("[build] WARNING: no 'model' key in checkpoint. Detector starts from random weights.")

        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    # 1. trajectory modeling (currently, only FFNs are used):
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    ) if config["ONLY_DETR"] is False else None
    # 2. ID decoder:
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    ) if config["ONLY_DETR"] is False else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )

    return motip_model, detr_criterion
