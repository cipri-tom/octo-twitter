#!/bin/bash

usage="Usage: ./preprocess (test|train) path/to/input [path/to/output]"

if [ "$#" -lt 2 ]; then
  echo "$usage"
  exit 1
fi

input="$2"

if [ -z $3 ]; then
  output="${input%.*}"
  output="${output}_preprocess.txt"
else
  output=$3
fi

echo "Preprocessing $input as $1. Writing to $output."
if [ "$1" == "train" ]; then
  cat "$input" | ruby -n preprocess-twitter.rb > $output
elif [ "$1" == "test" ]; then
  cat "$input" | cut -d, -f2- | ruby -n preprocess-twitter.rb \
               | awk 'BEGIN {OFS=","} {print  NR, $0 }' > $output
else
  echo "$usage"
  exit 2
fi

