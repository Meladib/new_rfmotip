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
            reid_proj: nn.Module = None,        # NEW
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        self.reid_proj = reid_proj              # NEW
        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000
        return

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if "use_checkpoint" in kwargs:
                    out = checkpoint(
                        self.detr, frames,
                        use_reentrant=False,
                    )
                else:
                    out = self.detr(samples=frames)
                if self.reid_proj is not None:
                    out["outputs"] = self.reid_proj(out["outputs"].detach())
                return out
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")
