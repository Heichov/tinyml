# scale the baseline networks until it fits the memory constraints
import sys

sys.path.append(".")
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='mbv2', choices=['mbv2', 'proxyless'])
parser.add_argument('--n_class', type=int, default=1000)
parser.add_argument('--engine', type=str, default='tinyengine', choices=['tflite', 'tinyengine'])
parser.add_argument('--sram_limit', type=int, default=320, help='SRAM limit in kB')
parser.add_argument('--flash_limit', type=int, default=1024, help='Flash limit in kB')
args = parser.parse_args()

# width_mult_candidate = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
width_mult_candidate = [round(w, 2) for w in np.linspace(0.1, 1, int((1-0.1)/0.05+1))]  # more fine-graned width mult
resolution_candidate = list(range(48, 192 + 1, 16))

if args.engine == 'tflite':  # suppress warnings
    import tensorflow as tf

    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def modify_config_width(cfg, width_mult=None, n_class=None, divisible_by=8):
    def modify_width(x):
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        import numpy as np
        return int(np.ceil(x * 1. * width_mult / divisible_by) * divisible_by)

    import copy
    cfg = copy.deepcopy(cfg)

    if width_mult is not None:  # use width multiplier
        cfg['first_conv']['out_channels'] = modify_width(cfg['first_conv']['out_channels'])

        cfg['feature_mix_layer']['in_channels'] = modify_width(cfg['feature_mix_layer']['in_channels'])
        cfg['feature_mix_layer']['out_channels'] = modify_width(cfg['feature_mix_layer']['out_channels'])

        cfg['classifier']['in_features'] = modify_width(cfg['classifier']['in_features'])
        if n_class is not None:
            cfg['classifier']['out_features'] = n_class

        for blk in cfg['blocks']:
            if 'in_channels' in blk['mobile_inverted_conv']:
                blk['mobile_inverted_conv']['in_channels'] = modify_width(blk['mobile_inverted_conv']['in_channels'])
                blk['mobile_inverted_conv']['out_channels'] = modify_width(blk['mobile_inverted_conv']['out_channels'])
            if blk['shortcut'] is not None:
                blk['shortcut']['in_channels'] = modify_width(blk['shortcut']['in_channels'])
                blk['shortcut']['out_channels'] = modify_width(blk['shortcut']['out_channels'])

    return cfg


def scale_baseline_network():
    import json
    from tinynas.nn.networks import ProxylessNASNets, MobileNetV2
    legal_configs = []

    for width_mult in width_mult_candidate:
        for resolution in resolution_candidate[::-1]:

            print('w{}-r{}'.format(width_mult, resolution))
            if args.arch == 'mbv2':
                model = MobileNetV2(n_classes=args.n_class, width_mult=width_mult, disable_keep_last_channel=True)
            elif args.arch == 'proxyless':
                org_config = json.load(open('./assets/configs/proxyless_mobile.json'))
                cfg = modify_config_width(org_config, width_mult, n_class=args.n_class)
                model = ProxylessNASNets.build_from_config(cfg)
            else:
                raise NotImplementedError

            model.eval()

            if args.engine == 'tinyengine':  # evaluate with json IR
                from tinyengine.memory_profiler import profile_tinyengine
                this_sram, this_flash = profile_tinyengine(model, data_shape=(1, 3, resolution, resolution))
            elif args.engine == 'tflite':
                from tinyengine.memory_profiler import profile_tflite
                this_sram, this_flash = profile_tflite(model, data_shape=(1, 3, resolution, resolution))
            else:
                raise NotImplementedError

            # if the flash size overflows, lower then resolution will not help
            if this_flash / 1024. > args.flash_limit:
                print(' * model too large... skip the width-mult')
                break

            if this_sram / 1024. < args.sram_limit and this_flash / 1024. < args.flash_limit:
                print('*' * 10, 'Found valid config: w{}-r{} (sram: {}kB, flash: {}kB)'.format(width_mult, resolution,
                                                                                               int(this_sram / 1024),
                                                                                               int(this_flash / 1024)))
                legal_configs.append((width_mult, resolution))
                break  # found the largest feasible resolution for the current model size

    idx2dispose = []
    for i in range(len(legal_configs)):
        for j in range(i+1, len(legal_configs)):
            if legal_configs[j][0] > legal_configs[i][0] and legal_configs[j][1] >= legal_configs[i][1]:
                idx2dispose.append(i)
                break
    legal_configs = [c for i_c, c in enumerate(legal_configs) if i_c not in idx2dispose]

    print(' * All legal configs:')
    for c in legal_configs:
        print('w{}-r{}'.format(c[0], c[1]))



if __name__ == '__main__':
    scale_baseline_network()
