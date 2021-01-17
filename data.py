import argparse
import csv
import os

parser = argparse.ArgumentParser()

parser.add_argument('input_dir')
parser.add_argument('output_dir')

parser.add_argument('dataset', choices=['dd', 'ed'])

parser.add_argument('--context', action='store_true')
parser.add_argument('--sep_token', default='</s>')

args = parser.parse_args()

for split in ['train', 'validation', 'test']:
    src_texts = []
    tgt_texts = []

    # DailyDialog
    if args.dataset == 'dd':
        with open(os.path.join(args.input_dir, split, 'dialogues_' + split + '.txt')) as f:
            for line in f:
                utterances = [utterance.strip() for utterance in line.split('__eou__')[:-1]]
                for i in range(1, len(utterances)):
                    src_texts.append(args.sep_token.join(utterances[:i]) if args.context else utterances[i-1])
                    tgt_texts.append(utterances[i])

    # EmpatheticDialogues
    else:
        with open(os.path.join(args.input_dir, split[:5] + '.csv')) as f:
            utterances = []
            conv_id = ''
            for row in csv.DictReader(f):
                if row['conv_id'] != conv_id and utterances:
                    for i in range(1, len(utterances)):
                        src_texts.append(args.sep_token.join(utterances[:i]) if args.context else utterances[i-1])
                        tgt_texts.append(utterances[i])
                    utterances = []
                utterances.append(row['utterance'].replace('_comma_', ','))
                conv_id = row['conv_id']

    with open(os.path.join(args.output_dir, ('val' if split == 'validation' else split) + '.tsv'), 'w') as f:
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            f.write(src_text + '\t' + tgt_text + '\n')
