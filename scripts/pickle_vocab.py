#!/usr/bin/env python3
import pickle
import sys

def main():
    in_path, out_path = '../data/vocab_cut.txt', '../data/vocab.pkl'
    if len(sys.argv) >= 2:
        in_path  = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    print("Reading from %s, writing to %s" % (in_path, out_path))

    vocab = dict()
    with open(in_path) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(out_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
