"""
diagnostics/diag_build_utils.py
================================
Shared model-building utility for RF-MOTIP diagnostic scripts.

WHY THIS EXISTS — two problems solved here:

Problem 1 — MultiScaleDeformableAttention import error:
  models/motip/__init__.py has an unconditional top-level import:
      from models.deformable_detr import build_deformable_detr
  This triggers the CUDA extension at import time regardless of DETR_FRAMEWORK.
  Fix: bypass models/motip/__init__.py entirely, build components directly.

Problem 2 — KeyError on DETR_HIDDEN_DIM:
  The config system uses a super-config inheritance pattern:
    configs/rf_detr_motip_dancetrack.yaml
      └── SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
  rf_detr_motip_dancetrack.yaml only contains OVERRIDES — it inherits all
  architecture keys (DETR_HIDDEN_DIM, FFN_DIM_RATIO, HEAD_DIM, REL_PE_LENGTH,
  USE_AUX_LOSS, USE_SHARED_AUX_HEAD, etc.) from the base config.
  Diagnostic scripts that call yaml.safe_load() on only one file miss all
  inherited keys.
  Fix: load_config() reads SUPER_CONFIG_PATH and merges — base first,
  then overrides applied on top. This mirrors what the training code does.

USAGE:
    from diagnostics.diag_build_utils import build_rf_motip, build_tracker, load_config

    CONFIG_PATH = "configs/rf_detr_motip_dancetrack.yaml"   # RF-DETR override config
    config  = load_config(CONFIG_PATH)                       # merges super-config automatically
    model   = build_rf_motip(config, checkpoint_path, device)
    tracker = build_tracker(model, config)
"""

import os
import torch


# ---------------------------------------------------------------------------
# Config loader with super-config merging
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Load a YAML config, recursively merging SUPER_CONFIG_PATH if present.

    Merge order:
      1. Load the super config (base values)
      2. Load this config (overrides)
      3. Return merged dict (this config wins on conflict)

    This mirrors the training code's config inheritance pattern.
    The rf_detr config sets SUPER_CONFIG_PATH → r50_deformable_detr_motip_dancetrack.yaml,
    which contains DETR_HIDDEN_DIM, FFN_DIM_RATIO, HEAD_DIM, REL_PE_LENGTH, etc.
    """
    import yaml

    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    super_path = config.get("SUPER_CONFIG_PATH", None)
    if super_path:
        # Resolve relative to the location of this config file
        base_dir   = os.path.dirname(config_path)
        
        # Resolve super_path relative to repo root first, then fall back to
        # relative-to-this-config. Handles both "configs/foo.yaml" and "./configs/foo.yaml".
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        super_abs = os.path.join(repo_root, super_path)
        if not os.path.exists(super_abs):
            super_abs = os.path.abspath(os.path.join(base_dir, super_path))
        if not os.path.exists(super_abs):
            raise FileNotFoundError(
                f"Cannot resolve SUPER_CONFIG_PATH '{super_path}' from either "
                f"repo root ({repo_root}) or config dir ({base_dir})"
            )
        
        super_config = load_config(super_abs)   # recursive — handles chains
        # Merge: super_config is the base, current config overrides
        merged = {**super_config, **config}
        merged.pop("SUPER_CONFIG_PATH", None)   # clean up — no longer needed
        return merged

    return config


# ---------------------------------------------------------------------------
# RF-MOTIP model builder (bypasses models/motip/__init__.py)
# ---------------------------------------------------------------------------

def build_rf_motip(config: dict, checkpoint_path: str, device: torch.device):
    """
    Builds the MOTIP model using only the RF-DETR path.

    Does NOT import models.motip (which triggers MultiScaleDeformableAttention).
    All architecture hyperparameters come from config after super-config merge.

    Architecture values in the merged config (from base + rf_detr override):
      DETR_HIDDEN_DIM:      256   (from base config, used as detr_dim)
      FEATURE_DIM:          256
      ID_DIM:               256
      FFN_DIM_RATIO:        2
      HEAD_DIM:             32
      NUM_ID_DECODER_LAYERS: 6
      NUM_ID_VOCABULARY:    50
      REL_PE_LENGTH:        30
      USE_AUX_LOSS:         False
      USE_SHARED_AUX_HEAD:  True
    """
    from models.rfdetr.models.lwdetr import build_model
    from models.motip.trajectory_modeling import TrajectoryModeling
    from models.motip.id_decoder import IDDecoder
    from models.motip.motip import MOTIP

    # --- Validate all required keys before touching the model ---
    required = [
        "CKPT_PATH", "DETR_HIDDEN_DIM", "FFN_DIM_RATIO", "FEATURE_DIM",
        "ID_DIM", "HEAD_DIM", "NUM_ID_DECODER_LAYERS", "NUM_ID_VOCABULARY",
        "REL_PE_LENGTH", "USE_AUX_LOSS", "USE_SHARED_AUX_HEAD",
    ]
    missing_keys = [k for k in required if k not in config]
    if missing_keys:
        raise KeyError(
            f"Config is missing keys after super-config merge: {missing_keys}\n"
            f"Make sure you are loading 'configs/rf_detr_motip_dancetrack.yaml' "
            f"(the RF-DETR override), NOT 'configs/r50_deformable_detr_motip_dancetrack.yaml' "
            f"(the base). The load_config() function in diag_build_utils.py handles the "
            f"SUPER_CONFIG_PATH merge automatically when given the override config."
        )

    # --- 1. RF-DETR detector ---
    det_ckpt_path = config["CKPT_PATH"]
    if not os.path.exists(det_ckpt_path):
        raise FileNotFoundError(
            f"RF-DETR detector checkpoint not found: {det_ckpt_path}\n"
            f"Expected at CKPT_PATH from config."
        )

    det_ckpt  = torch.load(det_ckpt_path, map_location="cpu", weights_only=False)
    args_ckpt = det_ckpt["args"]

    # Replicate the class-count adjustment from models/motip/__init__.py
    args_ckpt.num_classes -= 1
    detr = build_model(args=args_ckpt)
    args_ckpt.num_classes += 1

    # Load detector weights, filtering shape-mismatched class-head keys
    det_model_state = det_ckpt.get("model", None)
    if det_model_state is not None:
        model_state = detr.state_dict()
        filtered = {}
        for k, v in det_model_state.items():
            bare_k = k[5:] if k.startswith("detr.") else k
            if bare_k in model_state and v.shape == model_state[bare_k].shape:
                filtered[bare_k] = v
        missing, unexpected = detr.load_state_dict(filtered, strict=False)
        print(f"  [RF-DETR] loaded {len(filtered)} weights, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("  [RF-DETR] WARNING: no 'model' key in detector checkpoint — random weights")

    # --- 2. TrajectoryModeling ---
    trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],       # 256
        ffn_dim_ratio=config["FFN_DIM_RATIO"],     # 2
        feature_dim=config["FEATURE_DIM"],         # 256
    )

    # --- 3. IDDecoder ---
    id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],                # 256
        id_dim=config["ID_DIM"],                          # 256
        ffn_dim_ratio=config["FFN_DIM_RATIO"],            # 2
        num_layers=config["NUM_ID_DECODER_LAYERS"],       # 6
        head_dim=config["HEAD_DIM"],                      # 32
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],    # 50
        rel_pe_length=config["REL_PE_LENGTH"],            # 30
        use_aux_loss=config["USE_AUX_LOSS"],              # False
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"], # True
    )

    # --- 4. MOTIP wrapper ---
    model = MOTIP(
        detr=detr,
        detr_framework="rf_detr",
        only_detr=False,
        trajectory_modeling=trajectory_modeling,
        id_decoder=id_decoder,
    )

    # --- 5. Load MOTIP checkpoint (TrajectoryModeling + IDDecoder weights) ---
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MOTIP checkpoint not found: {checkpoint_path}")

    motip_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state      = motip_ckpt.get("model", motip_ckpt)
    state      = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Report what was loaded — explicit so any mismatch is immediately visible
    traj_n  = sum(1 for k in state if "trajectory_modeling" in k)
    id_n    = sum(1 for k in state if "id_decoder" in k)
    detr_n  = sum(1 for k in state if "detr" in k)
    print(f"  [MOTIP ckpt] trajectory_modeling={traj_n} keys, "
          f"id_decoder={id_n} keys, detr={detr_n} keys")

    non_detr_missing = [k for k in missing if "detr" not in k]
    if non_detr_missing:
        print(f"  [MOTIP ckpt] WARNING — non-DETR missing keys: {non_detr_missing[:5]}")

    model.eval()
    model.to(device)
    print(f"  Model ready on {device}.")
    return model


# ---------------------------------------------------------------------------
# RuntimeTracker builder
# ---------------------------------------------------------------------------

def build_tracker(model, config: dict, sequence_hw: tuple):
    """
    sequence_hw: (height, width) of the current sequence — read from seqinfo.ini.
    RuntimeTracker derives num_id_vocabulary from the model itself.
    Must be called per-sequence since sequence_hw changes.
    """
    from models.runtime_tracker import RuntimeTracker

    dtype_str = config.get("INFERENCE_DTYPE", "FP32").upper()
    dtype = torch.float16 if dtype_str == "FP16" else torch.float32

    tracker = RuntimeTracker(
        model=model,
        sequence_hw=sequence_hw,
        use_sigmoid=config.get("USE_SIGMOID", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "object-max"),
        miss_tolerance=config.get("MISS_TOLERANCE", 30),
        det_thresh=config.get("DET_THRESH", 0.3),
        newborn_thresh=config.get("NEWBORN_THRESH", 0.6),
        id_thresh=config.get("ID_THRESH", 0.2),
        area_thresh=config.get("AREA_THRESH", 0),
        only_detr=False,
        dtype=dtype,
    )
    return tracker