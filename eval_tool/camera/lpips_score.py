import os
import pathlib
from PIL import Image
from tqdm import tqdm

import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch.nn.functional as F
from torchvision import transforms
import lpips

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path_target', type=str, default=None,
                    help='Path to target patches')
parser.add_argument('--path_pred', type=str, default=None,
                    help='Path to predicted patches')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathsDataset(torch.utils.data.Dataset):
    def __init__(self, path_target, path_pred, transforms):
        self.files_target = self.get_files(path_target)
        self.files_pred = self.get_files(path_pred)
        assert len(self.files_target) == len(self.files_pred), 'Number of reference and predicted images should be same'
        self.transforms = transforms

    def get_files(self, path):
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        return files

    def __len__(self):
        return len(self.files_target)

    def __getitem__(self, i):
        patch_target = self.transforms(Image.open(self.files_target[i]))
        patch_pred = self.transforms(Image.open(self.files_pred[i]))

        return patch_target, patch_pred

@torch.no_grad()
def calculate_lpips_score_given_paths(path_target, path_pred, batch_size, device, num_workers):
    lpips_model = lpips.LPIPS(net='alex').to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImagePathsDataset(path_target, path_pred, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    lpips_list = []
    for patch_target, patch_pred in tqdm(dataloader):
        patch_target = patch_target.to(device)
        patch_pred = patch_pred.to(device)

        lpips_list.append(lpips_model(patch_target, patch_pred))

    lpips_list = torch.cat(lpips_list, dim=0)

    return lpips_list.mean().item()


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

    lpips_value = calculate_lpips_score_given_paths(
        args.path_target, args.path_pred, args.batch_size, device, num_workers
    )

    print('LPIPS: ', lpips_value)


if __name__ == '__main__':
    main()