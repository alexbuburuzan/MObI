"""
Inspired from:
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models
"""

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from scipy import linalg

from rangenet.model import Model as rangenet

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path-target', type=str, default=None,
                    help='Path to target range views')
parser.add_argument('--path-pred', type=str, default=None,
                    help='Path to edited range views')

class RangePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        depth_interval = (1.4, 54),
    ):
        path = pathlib.Path(path)
        self.files = list(path.glob('*.npy'))
        self.depth_interval = depth_interval

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file_path = self.files[i]
        range_data = np.load(file_path)
        range_depth, range_int, pitch, yaw = range_data[0], range_data[1], range_data[2], range_data[3]

        # unnorm range depth
        range_depth = (range_depth + 1) / 2 * self.depth_interval[1]
        valid_mask = np.logical_and(range_depth > self.depth_interval[0], range_depth < self.depth_interval[1])

        range_xyz = -np.ones((3, *range_depth.shape))
        range_xyz[0] = np.cos(yaw) * np.cos(pitch) * range_depth
        range_xyz[1] = -np.sin(yaw) * np.cos(pitch) * range_depth
        range_xyz[2] = np.sin(pitch) * range_depth

        range_data = np.concatenate(
            [range_depth[None, :], range_int[None, :], range_xyz],
            axis=0
        )

        range_data[:, ~valid_mask] = -1
        range_data = torch.tensor(range_data).float()
        range_data = F.interpolate(range_data.unsqueeze(1), size=(64, 1024), mode='nearest').squeeze()

        return range_data


def get_activations_of_path(path, model, batch_size=50, device='cpu', num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- path        : Path to range views
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    print(path)
    dataset = RangePathDataset(path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), 512))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch, return_final_logits=True, agg_type="depth")

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def compute_statistics_of_path(path, model, batch_size=64,
                               device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- path        : Path to files
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_of_path(path, model, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def build_model(model_folder, device):
    config = yaml.safe_load(open(os.path.join(model_folder, 'config.yaml'), 'r'))
    model = rangenet(config)
    model.load_pretrained_weights(model_folder)

    model.to(device)
    model.eval()

    return model

def calculate_frid_given_paths(path_target, path_pred, batch_size, device, num_workers=1):
    model = build_model("./eval_tool/lidar/rangenet", device=device)

    m1, s1 = compute_statistics_of_path(path_target, model, batch_size,
                                        device, num_workers)
    m2, s2 = compute_statistics_of_path(path_pred, model, batch_size,
                                        device, num_workers)
    frid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return frid_value

def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    score = calculate_frid_given_paths(
        args.path_target, args.path_pred, args.batch_size,
        device, num_workers
    )

    print("FRD: ", score)

if __name__ == "__main__":
    main()