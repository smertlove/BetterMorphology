

UNDEFINED = "[UNDEFINED]"

UPOSES = (
    UNDEFINED,
    'ADJ',
    'ADP',
    'ADV',
    'ADVPRO',
    'ANUM',
    'AUX',
    'CCONJ',
    'COM',
    'DET',
    'INIT',
    'INTJ',
    'NOUN',
    'NUM',
    'PARENTH',
    'PART',
    'PRED',
    'PREDPRO',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X'
)

UPOS2ID = {upos: idx for idx, upos in enumerate(UPOSES)}
ID2UPOS = {idx: upos for upos, idx in UPOS2ID.items()}
