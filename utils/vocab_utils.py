def parse_vocab(vocab_file):
    vocabs = []
    with open(vocab_file, mode='rt', encoding='utf8', buffering=8192) as f:
        for line in f:
            line = line.strip(' ').strip('\n')
            if not line:
                continue
            vocabs.append(line)
    return vocabs