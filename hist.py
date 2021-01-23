import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
from transformers import BartTokenizer, T5Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='dd', choices=['dd', 'ed'])
parser.add_argument('--tokenizer', default='bart', choices=['bart', 't5'])

args = parser.parse_args()

tokenizer = (
    T5Tokenizer.from_pretrained('t5-large') if args.tokenizer == 't5'
    else BartTokenizer.from_pretrained('facebook/bart-large')
)

fig, ax = plt.subplots()

bins = 50
range_ = (0, 200)
alpha = 0.5

for split in ['bi', 'multi']:
    texts = []
    with open(os.path.join(args.dataset + '_' + split, 'train.tsv')) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            texts.append(row[0])

    x = [len(input_ids) for input_ids in tokenizer(texts)['input_ids']]    
    label = split + '-turn'

    ax.hist(x, bins, range_, label=label, alpha=alpha)

ax.set_xlabel('Length')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Length')

ax.legend()

plt.tight_layout()
plt.savefig('hist_' + args.dataset + '_' + args.tokenizer + '.png')
