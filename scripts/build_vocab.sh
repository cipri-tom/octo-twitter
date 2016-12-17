#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
echo 'Usage: build_vocab [path/to/pos] [path/to/neg] [path/to/vocab]'

if [ -z $1 ]; then
  pos="../data/train_pos.txt"
else
  pos=$1
fi

if [ -z $2 ]; then
  neg="../data/train_neg.txt"
else
  neg=$2
fi

if [ -z $3 ]; then
  out="../data/vocab.txt"
else
  out=$3
fi

echo "Using paths: ($pos, $neg) -> $out"

cat "$pos" "$neg" | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > "$out"
