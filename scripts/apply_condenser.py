"""
Apply an already-trained condenser (see train_condensed_embedding.py) to a
new raw foundation-model embedding file, producing a condensed embedding for
every station in that file -- no streamflow target needed.

The condenser's encoder never saw streamflow or forcings during training,
only the raw embedding (see train_condensed_embedding.py's module
docstring), so it generalizes to any station set with the same embedding
dimension, gauged or not. This script is the pure-inference counterpart:
load a checkpoint train_condensed_embedding.py produced, run its encoder
over a (possibly much larger, possibly ungauged) embedding file, and write
the result in the same station_ids x time x embed_dim NetCDF format
generate_embeddings.py / train_condensed_embedding.py's --export_embedding
already use.

Usage
-----
python scripts/apply_condenser.py \\
    --condenser /path/to/MyDataset_condenser_monthly64.pt \\
    --embedding_nc /path/to/raw_embeddings_monthly.nc \\
    --out /path/to/condensed_embedding_monthly.nc \\
    --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from train_condensed_embedding import Condenser, open_embedding  # noqa: E402
from generate_embeddings import save_embedding_nc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--condenser', required=True, help="Checkpoint from train_condensed_embedding.py.")
    p.add_argument('--embedding_nc', required=True, help="Raw foundation-model embedding file to condense.")
    p.add_argument('--embedding_var_name', default='embedding')
    p.add_argument('--out', required=True, help="Output condensed-embedding NetCDF path.")
    p.add_argument('--batch_size', type=int, default=256, help="Stations per forward pass.")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.condenser, map_location=device)
    condenser = Condenser(ckpt['d_in'], ckpt['width']).to(device)
    condenser.encoder.load_state_dict(ckpt['encoder_state_dict'])
    condenser.eval()
    mode = ckpt.get('mode', 'supervised')  # checkpoints predating recon-only mode were all supervised
    print(f"Loaded {mode} condenser: {ckpt['d_in']} -> {ckpt['width']} "
          f"(trained on {ckpt.get('source_embedding_nc', '?')}"
          + (f" for task {ckpt.get('task_nc', '?')} / target '{ckpt.get('target_var', '?')}')"
             if mode == 'supervised' else ", reconstruction loss only)"))

    print(f"Opening embedding file {args.embedding_nc} ...")
    emb_ds, emb_var, emb_ids, emb_dates = open_embedding(args.embedding_nc, args.embedding_var_name)
    d_in = emb_var.shape[-1]
    if d_in != ckpt['d_in']:
        raise ValueError(f"Embedding dim {d_in} != condenser's expected input dim {ckpt['d_in']}")

    n_stations, n_time = emb_var.shape[0], emb_var.shape[1]
    print(f"  {n_stations} stations, {n_time} timesteps, d_in={d_in}")

    condensed = np.zeros((n_stations, n_time, ckpt['width']), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n_stations, args.batch_size):
            s1 = min(n_stations, s0 + args.batch_size)
            raw = np.asarray(emb_var[s0:s1, :, :], dtype=np.float32)
            raw = np.nan_to_num(raw, nan=0.0)
            z, _ = condenser(torch.from_numpy(raw).to(device))
            condensed[s0:s1] = z.cpu().numpy()
            print(f"  {s1}/{n_stations} stations condensed")

    save_embedding_nc(
        args.out,
        station_ids=emb_ids,
        lat=None, lon=None,
        dates=emb_dates,
        emb_tnd=np.transpose(condensed, (1, 0, 2)),  # [T, N, D] time-major, matching save_embedding_nc's expectation
        var_name='embedding',
        complevel=4,
    )
    emb_ds.close()


if __name__ == '__main__':
    main()
