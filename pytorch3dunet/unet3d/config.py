import argparse

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config(train):
    parser = argparse.ArgumentParser(description='UNet3D')
    if train:
        parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False,
                            default='../resources/train_reg.yaml')
    else:
        parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False,
                            default='../resources/3DUnet_lightsheet_nuclei/test_config.yaml')


    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
