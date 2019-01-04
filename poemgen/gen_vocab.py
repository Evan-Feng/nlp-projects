import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/train.txt', help='training data')
    parser.add_argument('--output', default='./data/vocab.txt', help='output file')
    parser.add_argument('--encoding', default='utf-8', help='encoding format')
    args = parser.parse_args()

    with open(args.input, 'r', encoding=args.encoding) as fin:
        lines = [row.strip() for row in fin if len(row.strip()) > 0]

    vocab = []
    seen = set()
    for line in lines:
        for c in line.replace(' ', '').replace(',', '').replace('.', ''):
            if c not in seen:
                seen.add(c)
                vocab.append(c)

    with open(args.output, 'w', encoding=args.encoding) as fout:
        fout.write('\n'.join(vocab) + '\n')


if __name__ == '__main__':
    main()
