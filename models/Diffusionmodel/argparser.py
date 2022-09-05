import argparse as argp
def argparse():
    parser = argp.ArgumentParser(
        description='Train the DCASE2020 FUSS baseline source separation model.')
    parser.add_argument(
        '-tdd', '--train_data_dir', help='Data directory.',
        required=True)
    parser.add_argument(
        '-edd', '--eval_data_dir', help='Data directory.',
        required=True)
    parser.add_argument(
        '-en', '--exec_name', help='Directory for checkpoints and summaries.',
        required=True)
    parser.add_argument(
        '-rt', '--root', help='Directory for checkpoints and summaries.',
        required=True)
    parser.add_argument(
        '-dp', '--datapath', help='Directory for checkpoints and summaries.',
        required=True)
    args = parser.parse_args()
    return args