#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
echo 'Usage: cut_vocab [path/to/vocab] [path/to/vocab_cut]'

if [ -z $1 ]; then
  input="../data/vocab.txt"
else
  input=$1
fi

if [ -z $2 ]; then
  output="../data/vocab_cut.txt"
else
  output=$2
fi

echo "Using paths: $input, $output"

cat "$input" | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > "$output"
