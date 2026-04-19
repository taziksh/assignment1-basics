import argparse


def model_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("model")
    g.add_argument("--vocab-size", type=int, default=10000)
    g.add_argument("--context-length", type=int, default=256)
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument("--num-heads", type=int, default=16)
    g.add_argument("--d-ff", type=int, default=1344)
    g.add_argument("--rope-theta", type=float, default=10000)
    return p


def optim_parser():
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("optim")
    g.add_argument("--lr", type=float, default=0.001)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--beta-1", type=float, default=0.9)
    g.add_argument("--beta-2", type=float, default=0.999)
    g.add_argument("--eps", type=float, default=10e-6)
    return p
