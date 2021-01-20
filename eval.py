import csv
import os
import sys

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu


def corpus_dist(corpus, n):
    ngrams_set = set()
    num_tokens = 0
    for tokens in corpus:
        ngrams_set |= set(ngrams(tokens, n))
        num_tokens += len(tokens)
    return len(ngrams_set) / num_tokens


data_dir, output_dir = sys.argv[1:]

list_of_references = []
with open(os.path.join(data_dir, 'test.tsv')) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        list_of_references.append([word_tokenize(row[1])])

hypotheses = []
with open(os.path.join(output_dir, 'preds.txt')) as f:
    for line in f:
        hypotheses.append(word_tokenize(line))

# bleu
print('BLEU', corpus_bleu(list_of_references, hypotheses), sep='\t')

# distinct
for n in range(1, 3):
    print('distinct-' + str(n), corpus_dist(hypotheses, n), sep='\t')

# average length
print('Avg Len', sum(len(tokens) for tokens in hypotheses) / len(hypotheses), sep='\t')
