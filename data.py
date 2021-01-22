import argparse
import csv
import os

parser = argparse.ArgumentParser()

parser.add_argument('input_dir')
parser.add_argument('output_dir')

parser.add_argument('--corpus_name', default='dd', choices=['dd', 'ed'])

parser.add_argument('--context', action='store_true')
parser.add_argument('--sep_token', default='</s>')
parser.add_argument('--one_sep_token', action='store_true')

args = parser.parse_args()

for split in ['train', 'validation', 'test']:
    dialogues = []

    # DailyDialog
    if args.corpus_name == 'dd':
        with open(os.path.join(args.input_dir, split, 'dialogues_' + split + '.txt')) as f:
            for line in f:
                dialogues.append([u.strip() for u in line.split('__eou__')[:-1]])

    # EmpatheticDialogues
    else:  
        with open(os.path.join(args.input_dir, split[:5] + '.csv')) as f:
            utterances = []
            conv_id = ''
            for row in csv.DictReader(f, quoting=csv.QUOTE_NONE):
                if row['conv_id'] != conv_id and utterances:
                    dialogues.append(utterances)
                    utterances = []
                utterances.append(row['utterance'].replace('_comma_', ','))
                conv_id = row['conv_id']
            dialogues.append(utterances)

    pairs = []
    for utterances in dialogues:
        for i in range(1, len(utterances)):
            if args.context and i > 1:
                if args.one_sep_token:
                    src_text = ' '.join(utterances[:i-1]) + args.sep_token + utterances[i-1]
                else:
                    src_text = args.sep_token.join(utterances[:i])
            else:
                src_text = utterances[i-1]
            pairs.append((src_text, utterances[i]))

    with open(os.path.join(args.output_dir, ('val' if split == 'validation' else split) + '.tsv'), 'w') as f:
        for pair in pairs:
            f.write('\t'.join(pair) + '\n')
