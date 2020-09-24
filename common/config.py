# coding=utf-8
import argparse

def update_args(args, ex_args):
    for k, v in ex_args.__dict__.items():
        setattr(args, k, v)

def _add_test_args(parser):
    parser.add_argument('--test_file', type=str, help='path to the test set file')
    parser.add_argument('--model_file', type=str, help='path to the model file')
    parser.add_argument('--beam_size', type=int, help='decoder beam size')
    parser.add_argument('--max_decode_step', type=int, help='maximum decode step')

def _add_model_args(parser):
    pass

def _add_train_args(parser):
    # arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--asdl_file', type=str, help='Path to ASDL grammar specification')
    parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    parser.add_argument('--save_to', type=str, help='save the model to')
    parser.add_argument('--train_file', type=str, help='path to the training target file')
    parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    parser.add_argument('--enc_hid_size', type=int, help='encoder hidden size')
    parser.add_argument('--src_emb_size', type=int, help='sentence embedding size')
    parser.add_argument('--field_emb_size', type=int, help='field embedding size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--max_epoch', type=int, help='max epoch')

    parser.add_argument('--clip_grad', type=float, default=10.0, help='clip grad to')
    parser.add_argument('--lr', type=float, default=.003, help='learning rate')

    parser.add_argument('--log_every', type=int, help='log every iter')
    parser.add_argument('--run_val_after', type=int, default=5, help='run validation after')
    parser.add_argument('--max_decode_step', type=int, help='maximum decode step')

def parse_args(mode):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--asdl_file', type=str, help='Path to ASDL grammar specification')

    # parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')

    # parser.add_argument('--train_file', type=str, help='path to the training target file')
    # parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    if mode == 'train':
        _add_train_args(parser)
    elif mode == 'test':
        _add_test_args(parser)
    else:
        raise RuntimeError('unknown mode')
    
    args = parser.parse_args()
    print(args)
    return args
