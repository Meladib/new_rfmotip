# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MOTIP(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
            reid_proj: nn.Module = None,
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        self.reid_proj = reid_proj
        return

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if "use_checkpoint" in kwargs:
                    return checkpoint(
                        self.detr, frames,
                        use_reentrant=False,
                    )
                else:
                    return self.detr(samples=frames)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                if self.reid_proj is not None:
                    for key in ("trajectory_features", "unknown_features"):
                        feat = seq_info[key]
                        seq_info[key] = self.reid_proj(
                            feat.reshape(-1, feat.shape[-1])
                        ).reshape(feat.shape)
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")
