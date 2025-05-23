import pathlib
from PIL import Image
import torch
import os
from tqdm import tqdm
import clip
from einops import rearrange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch.nn.functional as F

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path_ref', type=str, default=None,
                    help='Path to reference object images')
parser.add_argument('--path_pred', type=str, default=None,
                    help='Path to predicted object images')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathsDataset(torch.utils.data.Dataset):
    def __init__(self, path_ref, path_pred, transforms):
        self.files_ref = self.get_files(path_ref)
        self.files_pred = self.get_files(path_pred)
        assert len(self.files_ref) == len(self.files_pred), 'Number of reference and predicted images should be same'
        self.transforms = transforms

    def get_files(self, path):
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        return files

    def __len__(self):
        return len(self.files_ref)

    def __getitem__(self, i):
        object_ref = self.transforms(Image.open(self.files_ref[i]))
        object_pred = self.transforms(Image.open(self.files_pred[i]))

        return object_ref, object_pred

@torch.no_grad()
def calculate_clip_score_given_paths(path_ref, path_pred, batch_size, device, num_workers):
    clip_model, preprocess = clip.load('ViT-B/32', device=device)

    dataset = ImagePathsDataset(path_ref, path_pred, preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    clip_model.eval()
    clip_model.to(device)

    sim_list = []
    for object_ref, object_pred in tqdm(dataloader):
        object_ref = object_ref.to(device)
        object_pred = object_pred.to(device)

        ref_feat = clip_model.encode_image(object_ref)
        pred_feat = clip_model.encode_image(object_pred)

        sim = 100 * F.cosine_similarity(ref_feat, pred_feat, dim=-1)
        sim_list.append(sim)

    sim_list = torch.cat(sim_list, dim=0)

    return sim_list.mean().item()


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

    clip_value = calculate_clip_score_given_paths(
        args.path_ref, args.path_pred, args.batch_size, device, num_workers
    )

    print('CLIP: ', clip_value)


if __name__ == '__main__':
    main()