import sys
sys.path.insert(0, 'src/')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--build_model', help='trains the model', action="store_true")
parser.add_argument('--predict', help='create predictions', action="store_true")
parser.add_argument('--submission', help='prepare submission', action="store_true")
parser.add_argument('--disable_gpu', help='disable gpu in tensorflow', action="store_true")
args = parser.parse_args()

if args.disable_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

from src import model, mask_to_submission

if args.build_model:
    model.train_model()

if args.predict:
    model.predict()

if args.submission:
    mask_to_submission.make_submission()