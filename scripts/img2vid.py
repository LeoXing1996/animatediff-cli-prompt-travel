"""This script support generate configs and videos from single image input
"""
import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

from animatediff.settings import ModelConfig, get_model_config

CONFIG_ROOT = 'config/img2vid'
os.makedirs(CONFIG_ROOT, exist_ok=True)

TILE_ROOT = 'data/controlnet_image/{}'
TILE_TEMPLATE = 'data/controlnet_image/{}/controlnet_tile/0.png'

IPADATER_ROOT = 'data/ip_adapter/{}'
IPADATER_TEMPLATE = 'data/ip_adapter/{}/0.png'


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--img')
    parser.add_argument('--base-cfg', type=str)

    parser.add_argument('-H', default=512)
    parser.add_argument('-W', default=512)

    parser.add_argument('--tile-weight', type=float, default=0.1)
    parser.add_argument('--disable-tile', action='store_true')

    parser.add_argument('--ip-adapter-weight', type=float, default=0.3)
    parser.add_argument('--disable-ip-adapter', action='store_true')

    parser.add_argument('--strength', type=float, default=0.85)

    parser.add_argument('--save-name', type=str, help='save name for both config file and output video')

    parser.add_argument('--work-dir', default='img2vid')

    return parser.parse_args()


def main():
    args = get_args()

    assert osp.exists(args.img), 'input image not exists'
    img_open = Image.open(args.img)

    save_name = args.save_name
    work_dir = osp.join('output', args.work_dir)

    base_cfg_path = Path(args.base_cfg)
    base_cfg: ModelConfig = get_model_config(base_cfg_path)

    base_cfg.name = base_cfg.name + '_img2vid'

    enable_ip_adapter = not args.disable_ip_adapter
    ip_adapter_weight = args.ip_adapter_weight
    os.makedirs(IPADATER_ROOT.format(save_name), exist_ok=True)
    img_open.save(osp.join(IPADATER_TEMPLATE.format(save_name)))
    base_cfg.ip_adapter_map['input_image_dir'] = IPADATER_TEMPLATE.format(save_name)
    base_cfg.ip_adapter_map['enable'] = enable_ip_adapter
    base_cfg.ip_adapter_map['scale'] = ip_adapter_weight

    enable_tile = not args.disable_tile
    tile_weight = args.tile_weight
    os.makedirs(TILE_ROOT.format(save_name), exist_ok=True)
    img_open.save(osp.join(IPADATER_TEMPLATE.format(save_name)))
    base_cfg.controlnet_map['input_image_dir'] = TILE_TEMPLATE.format(save_name)
    base_cfg.controlnet_map['controlnet_tile']['enable'] = enable_tile
    base_cfg.controlnet_map['controlnet_tile']['controlnet_conditioning_scale'] = tile_weight

    strength = args.strength

    cfg_save_dir = Path(CONFIG_ROOT) / (save_name + '.json')
    cfg_save_dir.write_text(base_cfg.json(indent=4), encoding='utf-8')

    img2vid_cmd = (f'animatediff generate -c {cfg_save_dir.absolute()}'
                   f' -H {args.H} -W {args.W} -L 16 -C 16 '
                   f' --img-path {args.img} --strength {strength} '
                   f' --save-name {save_name} --out-dir {work_dir}')
    print('img2vid cmd:')
    print(img2vid_cmd)
    os.system(img2vid_cmd)


if __name__ == '__main__':
    main()
