from common.config import *
from components.dataset import *

from grammar.grammar import Grammar

from grammar.sparql.sparql_transition_system import SparqlTransitionSystem
from models.ASN import ASNParser
from models import nn_utils

from torch import optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
import hiddenlayer as hl

def train(args):
    train_writer = SummaryWriter('./checkpoints/sparql/logs/train')
    valid_writer = SummaryWriter('./checkpoints/sparql/logs/val')

    train_set = Dataset.from_bin_file(args.train_file)
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])
    
    vocab = pickle.load(open(args.vocab, 'rb'))
    grammar = Grammar.from_text(open(args.asdl_file).read())
    # print(grammar)
    # transition_system = Registrable.by_name(args.transition_system)(grammar)
    transition_system = SparqlTransitionSystem(grammar)
    
    parser = ASNParser(args, transition_system, vocab)
    print(parser)
    # print(parser.parameters())
    nn_utils.glorot_init(parser.parameters())

    optimizer = optim.Adam(parser.parameters(), lr=args.lr)
    best_acc = 0.0
    log_every = args.log_every
    
    train_begin = time.time()
    for epoch in range(1, args.max_epoch + 1):
        train_iter = 0
        loss_val = 0.
        epoch_loss = 0.

        parser.train()

        epoch_begin = time.time()
        for batch_example in train_set.batch_iter(batch_size=args.batch_size, shuffle=False):
            # print(batch_example)
            optimizer.zero_grad()
            loss = parser.score(batch_example)
            loss_val += torch.sum(loss).data.item()
            epoch_loss += torch.sum(loss).data.item()
            loss = torch.mean(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), args.clip_grad)

            optimizer.step()
            train_iter += 1
            if train_iter % log_every == 0:
                print("[epoch {}, step {}] loss: {:.3f}".format(epoch, train_iter, loss_val / (log_every * args.batch_size )))
                loss_val = 0.

        # print(epoch, 'Train loss', '{:.3f}'.format(epoch_loss / len(train_set)), 'time elapsed %d' % (time.time() - epoch_begin))
        print('[epoch {}] train loss {:.3f}, epoch time {:.0f}, total time {:.0f}'.format(epoch, epoch_loss / len(train_set), time.time() - epoch_begin, time.time() - train_begin) )
        # train_writer.add_scalar('Loss', epoch_loss / len(train_set), epoch)
        if epoch > args.run_val_after:
            eval_begin = time.time()
            parser.eval()
            with torch.no_grad():
                parse_results = [parser.naive_parse(ex) for ex in dev_set]
            match_results = [transition_system.compare_ast(e.tgt_ast, r) for e, r in zip(dev_set, parse_results)]
            match_acc = sum(match_results) * 1. / len(match_results)
            # print('Eval Acc', match_acc)
            print('[epoch {}] eval acc {:.3f}, eval time {:.0f}'.format(epoch, match_acc, time.time() - eval_begin))
            # valid_writer.add_scalar('Acc', match_acc, epoch)
            
            if match_acc >= best_acc:
                best_acc = match_acc
                parser.save(args.save_to)

    # train_writer.add_graph(parser, dev_set, verbose=True)
    # transforms = [hl.transforms.Prune('Constant')]  # Removes Constant nodes from graph.
    # graph = hl.build_graph(parser, dev_set, transforms=transforms)
    # graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save('rnn_hiddenlayer', format='png')


if __name__ == '__main__':
    args = parse_args('train')
    train(args)
