"""Load the three-part simulation checkpoint for use with a guarded control loop."""
from pathlib import Path

import torch

from .actor import DirectIK, REF_LEAD_RAD, STEP_RAD
from .command import CommandCorrection, CommandSpacePolicyIK
from .holding import HoldingPrior, SmoothHolding


def load_sim(directory):
    directory = Path(directory)
    actor = DirectIK(directory / "actor.pt")
    workspace = CommandCorrection().eval()
    workspace.load_state_dict(torch.load(directory / "workspace.pt", map_location="cpu", weights_only=True))
    hold = HoldingPrior().eval()
    hold.load_state_dict(torch.load(directory / "hold_prior.pt", map_location="cpu", weights_only=True))
    command = CommandSpacePolicyIK(actor, workspace, max_step=STEP_RAD, max_lead=REF_LEAD_RAD)
    return SmoothHolding(command, hold)
