#!/usr/bin/env python

import torch
import argparse
import sys

def get_parser():
    parser = argparse.ArgumentParser(description='create state dict with only model weight information')
    parser.add_argument('--input', '-i', help='model.pth file from training')
    parser.add_argument('--output', '-o', default='model_deploy.pth', help='output model file (default: model_deploy.pth)')
    return parser
                            

if __name__ == '__main__':
    parser = get_parser()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    model_weights = torch.load(open(args.input, 'rb'))
    model_weights.pop('optimizer');
    model_weights.pop('scheduler');
    model_weights.pop('iteration');
    torch.save(model_weights, open(args.output, 'wb'))
