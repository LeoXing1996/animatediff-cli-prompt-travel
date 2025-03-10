"""
Adapted from https://github.com/huggingface/diffusers/issues/2326 by https://github.com/ignacfetser

The LoRA trained using Diffusers are saved in .bin or .pkl format, which must be converted to be used in Automatic1111 WebUI.

This script converts .bin or .pkl files into .safetensors format, which can be used in WebUI.

Put this file in the same folder of .bin or .pkl file and run `python convert-to-safetensors.py --file checkpoint_file`

"""
import re
import os
import argparse
import torch
from safetensors.torch import save_file
from safetensors import safe_open


def main(args):
    # use GPU or CPU
    device = 'cuda'
    checkpoint = {}
    with safe_open(args.file, framework='pt') as file:
        for k in file.keys():
            checkpoint[k] = file.get_tensor(k)
    # if torch.cuda.is_available():
    #     device = 'cuda'
    #     checkpoint = torch.load(args.file, map_location=torch.device('cuda'))
    # else:
    #     device = 'cpu'
    #     # if on CPU or want to have maximum precision on GPU, use default full-precision setting
    #     checkpoint = torch.load(args.file, map_location=torch.device('cpu'))

    print(f'device is {device}')

    new_dict = dict()
    for idx, key in enumerate(checkpoint):
        new_key = re.sub(r'\.processor\.', '_', key)
        new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
        new_key = re.sub('_lora.up.', '.lora_up.', new_key)
        new_key = re.sub('_lora.down.', '.lora_down.', new_key)
        new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
        new_key = re.sub('to_out', 'to_out_0', new_key)
        new_key = 'lora_unet_' + new_key
        new_key = new_key.replace('_unet.', '_', 1)
        print(new_key)
        new_dict[new_key] = checkpoint[key]

    file_name = os.path.splitext(args.file)[0]  # get the file name without the extension
    new_lora_name = file_name + '_converted.safetensors'
    print("Saving " + new_lora_name)
    save_file(new_dict, new_lora_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        required=True,
        help="path to the full file name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
