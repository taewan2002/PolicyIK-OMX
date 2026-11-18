"""Train the Sim-IK workspace and holding modules from simulator samples."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .command import CommandCorrection, scaled_error
from .holding import HoldingPrior
from .kinematics import forward


def load(archive, keys, device):
    with np.load(archive,allow_pickle=False) as data:
        arrays = tuple(np.asarray(data[key],dtype=np.float64) for key in keys)
    if not arrays[0].shape[0] or any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Empty or nonfinite training data")
    if keys == ("start","end"):
        q, target = arrays
        arrays = (q[:,:5], scaled_error(forward(target),forward(q)),target[:,:5]-q[:,:5])
    else:
        q, offset = arrays
        arrays = (q[:,:5],offset.clip(-.025,.025))
    return tuple(torch.as_tensor(values,dtype=torch.float32,device=device) for values in arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("workspace","hold"))
    parser.add_argument("samples",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--init",type=Path)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    workspace = args.kind=="workspace"
    torch.manual_seed(631717 if workspace else 112632)
    model = (CommandCorrection() if workspace else HoldingPrior()).to(args.device)
    if args.init:
        model.load_state_dict(torch.load(args.init,map_location="cpu",weights_only=True))
    x,*other,y = load(args.samples,("start","end") if workspace else ("command","deflection"),args.device)
    steps,batch = (3000,1024) if workspace else (850,256)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0015 if workspace else .003,
                                  weight_decay=.0001 if workspace else .01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.00012) if workspace else None
    for _ in range(steps):
        idx = torch.randint(len(x),(batch,),device=args.device)
        prediction = model(x[idx],*(values[idx] for values in other))
        if workspace:
            scale = y[idx].square().mean(1,keepdim=True).sqrt().clamp_min(.001)
            loss = F.smooth_l1_loss(prediction/scale,y[idx]/scale,beta=.2)
        else:
            loss = ((prediction-y[idx])/.025).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if schedule is not None:
            schedule.step()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.cpu().state_dict(),args.output)


if __name__=="__main__":
    main()
