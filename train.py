from common.config import *
from components.dataset import *

from grammar.grammar import Grammar

from grammar.sparql.sparql_transition_system import SparqlTransitionSystem
from models.ASN import ASNParser
from models import nn_utils

from torch import optim
from tqdm import tqdm
import time

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return

def train(args):

    train_set = Dataset.from_bin_file(args.train_file)
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else:
        dev_set = Dataset(examples=[])
    
    vocab = pickle.load(open(args.vocab, 'rb'))
    grammar = Grammar.from_text(open(args.asdl_file).read())

    transition_system = SparqlTransitionSystem(grammar)
    
    parser = ASNParser(args, transition_system, vocab)

    if args.cuda:
        parser = parser.cuda()

    nn_utils.glorot_init(parser.parameters())

    optimizer = optim.AdamW(parser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sch_step_size, gamma=args.gamma)
    best_acc = 0
    log_every = args.log_every
    
    train_begin = time.time()
    for epoch in range(1, args.max_epoch + 1):
        train_iter = 0
        loss_val = 0.
        epoch_loss = 0.

        parser.train()

        epoch_begin = time.time()
        for batch_example in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = parser.score(batch_example)
            loss_val += torch.sum(loss).data.item()
            epoch_loss += torch.sum(loss).data.item()
            loss = torch.mean(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), args.clip_grad)

            optimizer.step()
            lr_scheduler.step()
            train_iter += 1
            if train_iter % log_every == 0:
                print("[epoch {}, step {}] loss: {:.3f}".format(epoch, train_iter, loss_val / (log_every * args.batch_size)))
                loss_val = 0.


        print('[epoch {}] train loss {:.3f}, epoch time {:.0f}, total time {:.0f}, lr {:.5f}'.format(epoch, epoch_loss / len(train_set), time.time() - epoch_begin, time.time() - train_begin, get_lr(optimizer)))

        to_print = list()
        to_print_target = list()

        if epoch > args.run_val_after:
            for dev_set_ in tqdm(dev_set.batch_iter(batch_size=args.batch_size, shuffle=False)):
                eval_begin = time.time()
                parser.eval()
                with torch.no_grad():
                    batch = Batch(dev_set_, parser.grammar, parser.vocab, train=False, cuda=parser.args.cuda)
                    parse_results = list(zip(parser.naive_parse(batch), [dev_set_[ex].tgt_ast for ex in range(len(batch))]))

                    to_print.extend([transition_system.ast_to_surface_code(x[0]) for x in parse_results])
                    to_print_target.extend([transition_system.ast_to_surface_code(x[1]) for x in parse_results])

            # print(to_print)
            # print("-"*10)
            # print(to_print_target)

            match_results = [x == y for x, y in zip(to_print, to_print_target)]
            match_acc = sum(match_results) * 1. / len(match_results)

            print('[epoch {}] eval acc {:.3f}, eval time {:.0f}'.format(epoch, match_acc, time.time() - eval_begin))
            #
            if match_acc >= best_acc:
                best_acc = match_acc
                parser.save(args.save_to)


if __name__ == '__main__':
    args = parse_args('train')
    train(args)
