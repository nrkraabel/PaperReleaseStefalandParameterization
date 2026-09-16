"""
Train a linear condenser (embedding_size -> condensed_width) on one
resolution's embedding file from generate_embeddings.py, against
reconstruction loss. Full station set, full time range.

    python scripts/train_condensed_embedding.py \\
        --embedding_nc MyDataset_embeddings_daily.nc \\
        --condensed_width 64 \\
        --out condensers/MyDataset_condenser_daily64.pt \\
        --export_embedding
"""

import argparse
import os
import sys
import time

import netCDF4
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'src'), os.path.join(_ROOT, 'src', 'dmg'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_embeddings import save_embedding_nc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--embedding_nc', required=True)
    p.add_argument('--embedding_var_name', default='embedding')
    p.add_argument('--start_time', default=None, help="YYYY-MM-DD, subsets the embedding file.")
    p.add_argument('--end_time', default=None, help="YYYY-MM-DD.")
    p.add_argument('--condensed_width', type=int, required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64, help="Basins per minibatch.")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True)
    p.add_argument('--export_embedding', action='store_true',
                   help="Also write <out>_embedding.nc.")
    return p.parse_args()


class Condenser(nn.Module):
    def __init__(self, d_in, width):
        super().__init__()
        self.encoder = nn.Linear(d_in, width)
        self.decoder = nn.Linear(width, d_in)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def load(args):
    ds = netCDF4.Dataset(args.embedding_nc, 'r')
    var = ds[args.embedding_var_name]
    ids = np.array([str(s) for s in ds['station_ids'][:]])
    dates = pd.DatetimeIndex(netCDF4.num2date(
        ds['time'][:], units=ds['time'].units,
        calendar=getattr(ds['time'], 'calendar', 'standard'),
        only_use_cftime_datetimes=False))

    keep = np.ones(len(dates), dtype=bool)
    if args.start_time:
        keep &= dates >= pd.Timestamp(args.start_time)
    if args.end_time:
        keep &= dates <= pd.Timestamp(args.end_time)
    t_idx = np.nonzero(keep)[0]
    if len(t_idx) == 0:
        raise ValueError("--start_time/--end_time select no timesteps.")

    d_in = var.shape[-1]
    gb = len(ids) * len(t_idx) * d_in * 4 / 1024 ** 3
    print(f"  {len(ids)} stations, {len(t_idx)} timesteps, d_in={d_in}, ~{gb:.2f} GB")

    # Cached once; re-decompressing per minibatch dominates runtime at daily.
    cache = np.asarray(var[:, :, :], dtype=np.float32)[:, t_idx, :]
    ds.close()
    return np.nan_to_num(cache, nan=0.0), ids, dates[t_idx], d_in


def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"Opening {args.embedding_nc} ...")
    cache, ids, dates, d_in = load(args)
    n = len(ids)

    print(f"Condenser {d_in} -> {args.condensed_width} on {device}")
    model = Condenser(d_in, args.condensed_width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm, total, nb = rng.permutation(n), 0.0, 0
        for b0 in range(0, n, args.batch_size):
            x = torch.from_numpy(cache[perm[b0:b0 + args.batch_size]]).to(device).permute(1, 0, 2)
            loss = mse(model(x)[1], x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"  epoch {epoch:3d}: recon_mse={total / nb:.4f}  ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'encoder_state_dict': model.encoder.state_dict(), 'd_in': d_in,
                'width': args.condensed_width, 'source_embedding_nc': args.embedding_nc},
               args.out)
    print(f"Saved {args.out}")

    if not args.export_embedding:
        return
    print("Exporting condensed embedding ...")
    model.eval()
    out = np.zeros((n, cache.shape[1], args.condensed_width), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n, args.batch_size):
            idx = np.arange(s0, min(n, s0 + args.batch_size))
            out[idx] = model(torch.from_numpy(cache[idx]).to(device))[0].cpu().numpy()

    save_embedding_nc(
        os.path.splitext(args.out)[0] + '_embedding.nc',
        station_ids=np.asarray(ids), lat=None, lon=None, dates=dates,
        emb_tnd=np.transpose(out, (1, 0, 2)),  # save_embedding_nc wants time-major
        var_name='embedding', complevel=4,
    )


if __name__ == '__main__':
    main()
