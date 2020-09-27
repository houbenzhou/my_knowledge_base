import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('--in_file',
                        default='/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200923_gaofen_plane/epoch_24.pth',
                        help='input checkpoint filename')
    parser.add_argument('--out_file',
                        default='/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200923_gaofen_plane/out.pth',
                        help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()