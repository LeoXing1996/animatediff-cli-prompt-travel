import os
import os.path as osp
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from animatediff.settings import ModelConfig, get_model_config

BASE_CONFIG = 'configs/prompts/img2vid-base-cfg.json'
AD_CONFIG_ROOT = 'configs/benchmark'


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():

    args = get_args()

    MODEL_LIST = []  # TODO
    INP_IMG = []  # TODO
    PROMPT = []  # TODO
    N_PROMPT = []  # TODO

    WORK_DIR = './benchmark'
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(AD_CONFIG_ROOT, exist_ok=True)

    base_cfg_path = Path(BASE_CONFIG)
    base_cfg: ModelConfig = get_model_config(base_cfg_path)

    base_cfg.clip_skip = 1  # NOTE: set clip_skip as 1 since we do not have clip-skip in our repo

    for model, img, prompt, n_prompt in zip(MODEL_LIST, INP_IMG, PROMPT, N_PROMPT):
        config = deepcopy(base_cfg)

        config.path = model
        config.head_prompt = prompt
        config.n_prompt = [n_prompt]

        model_name = model_name.split('.')[0]

        config_save_path = osp.join(AD_CONFIG_ROOT, f'{model_name}.json')
        Path(config_save_path).write_text(config.json(indent=4), encoding='utf-8')
        print(f'Save base config for model {model} to {config_save_path}.')

        name = f'{model_name}_img_{img_name}'
        ad_cmd = (f'animatediff generate -c {Path(config_save_path).absolute()}'
                  f' -H {args.H} -W {args.W} -L 16 -C 16 '
                  f' --save-name {name}')
        srun_cmd = ('srun -p mm_lol -n1 -N1 --gres=gpu:1 --cpus-per-task 4 --async'
                    f'--job-name ad-{name} ' + run_cmd)
        if not args.dry_run:
            os.system(ad_cmd)
            time.sleep(1)
        print(f'command for vanilla animateDiff {name}')
        print(srun_cmd)

        # NOTE: maybe we do not need this?
        for enable_tile in [True, False]:
            for enable_ip_adapter in [True, False]:

                img_name = osp.basename(img).split('.')[0]
                name = f'{model_name}_img_{img_name}_tile_{enable_tile}_ip_{enable_ip_adapter}'

                run_cmd = (f'python scripts/img2vid.py --img {img} --base-cfg {config_save_path} '
                           f'--save-name {name} --work-dir {WORK_DIR}')
                if not enable_tile:
                    run_cmd += ' --disable-tile'
                if not enable_ip_adapter:
                    run_cmd += ' --disable-ip-adapter'

                srun_cmd = ('srun -p mm_lol -n1 -N1 --gres=gpu:1 --cpus-per-task 4 --async'
                            f'--job-name img-vid-{name} ' + run_cmd)
                if not args.dry_run:
                    os.system(srun_cmd)
                    time.sleep(1)
                print(f'command for {name}')
                print(srun_cmd)


if __name__ == '__main__':
    main()
