import argparse
from hswind_inference import config


def cmdline() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run Hs wind Sea inference on top of S1 L2 products',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--list_l2_nc',
        type=str,
        help='path to .txt file which list the absolut path(s) of L2 ocn measurements (.nc) to be treated',
        required=True
    )

    parser.add_argument(
        '-v', '--verbosity',
        type=str,
        help='Verbosity',
        default=config.INFERENCE.VERBOSITY,
        required=False
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Nr of Process to be launched',
        default=config.INFERENCE.BATCH_SIZE,
        required=False
    )
    parser.add_argument(
        '-o', '--outdir',
        type=str,
        help='Folder where the results will be written',
        default=config.filesystem.OUTPUT_FOLDER,
        required=False
    )

    parser.add_argument(
        '--aux_ml2',
        type=str,
        help='path to aux_ml2',
        required=True
    )

    parser.add_argument(
        '--log',
        help='To disable the main logging',
        action='store_true',
    )

    parser.add_argument(
        '--no_json',
        help='To return the full csv with the used features',
        action='store_true',
    )

    parser.add_argument(
        '--export_result',
        help='To export result',
        action='store_true',
    )

    return parser
