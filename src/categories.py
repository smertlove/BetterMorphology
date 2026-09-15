

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

UPOS2ID: dict[str, int] = {upos: idx for idx, upos in enumerate(UPOSES)}
ID2UPOS: dict[int, str] = {idx: upos for upos, idx in UPOS2ID.items()}
