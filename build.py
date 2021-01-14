import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('input_dir')
parser.add_argument('output_dir')

parser.add_argument('--context', action='store_true')
parser.add_argument('--sep_token', default='</s>')

args = parser.parse_args()

for split in ['train', 'validation', 'test']:
    src_texts = []
    tgt_texts = []
    with open(os.path.join(args.input_dir, split, 'dialogues_' + split + '.txt')) as f:
        for line in f:
            texts = [text.strip() for text in line.split('__eou__')[:-1]]
            for i in range(1, len(texts)):
                src_texts.append(args.sep_token.join(texts[:i]) if args.context else texts[i-1])
                tgt_texts.append(texts[i])
    
    with open(os.path.join(args.output_dir, ('val' if split == 'validation' else split) + '.tsv'), 'w') as f:
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            f.write(src_text + '\t' + tgt_text + '\n')
