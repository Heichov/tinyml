import sys
sys.path.append(".")

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()


if __name__ == '__main__':
    # 1. build the model
    from tinynas.nn.networks import ProxylessNASNets
    with open(args.config) as f:
        cfg = json.load(f)
    model = ProxylessNASNets.build_from_config(cfg)
    model.eval()
    resolution = cfg['resolution']
    data_shape = (1, 3, resolution, resolution)  # assume square input

    # 2. profile macs, params, and peak activation size
    # NOTE: the activation size is just a theoretical analysis, no including any temporary buffers,
    # which is much smaller compared to actual on-device measurement
    from utils import count_net_flops, count_parameters, count_peak_activation_size
    flops = count_net_flops(model, data_shape=data_shape)
    params = count_parameters(model)
    peak_act = count_peak_activation_size(model, data_shape=data_shape)

    print(' * #MACs:\t {:.3f}M'.format(flops / 1e6))
    print(' * #Params:\t {:.3f}M'.format(params / 1e6))
    print(' * PeakMem:\t {:.1f}kB (int8 mode)'.format(peak_act / 1024))
    print()

    # 3. profile the actual memory (SRAM/Flash) usage of different libraries
    # NOTE: the measurement might be slightly smaller compared to on-device measurement
    # the difference is <5kB for SRAM and <10kB for Flash
    # TF-Lite Micro
    import tensorflow as tf
    import os
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from tinyengine.memory_profiler import profile_tinyengine, profile_tflite

    print(' * TF-Lite Micro statistics:')
    sram, flash = profile_tflite(model, data_shape=data_shape)
    print(' * SRAM (TF-Lite):\t {:.1f}kB'.format(sram / 1024))
    print(' * Flash (TF-Lite):\t {:.1f}kB'.format(flash / 1024))
    print()

    # TinyEngine
    print(' * TinyEngine statistics:')
    sram, flash = profile_tinyengine(model, data_shape=data_shape)
    print(' * SRAM (TinyEngine):\t {:.1f}kB'.format(sram / 1024))
    print(' * Flash (TinyEngine):\t {:.1f}kB'.format(flash / 1024))
