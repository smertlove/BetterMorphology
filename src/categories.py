def get_id_mappings(*elements: str) -> tuple[dict[str, int], dict[int, str]]:
    elem2id = {elem: idx for idx, elem in enumerate(elements)}
    id2elem = {idx: elem for idx, elem in enumerate(elements)}
    return elem2id, id2elem


UNDEFINED = "[UND]"


UPOS2ID, ID2UPOS = get_id_mappings(
    "ADJ",
    "ADP",
    "ADV",
    "ADVPRO",  # not in GramEval2020
    "ANUM",  # not in GramEval2020
    "AUX",
    "CCONJ",
    "COM",  # not in GramEval2020
    "DET",
    "INIT",  # not in GramEval2020
    "INTJ",
    "NOUN",
    "NUM",
    "PARENTH",  # not in GramEval2020
    "PART",
    "PRED",  # not in GramEval2020
    "PREDPRO",  # not in GramEval2020
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)

ANIMACY2ID, ID2ANIMACY = get_id_mappings(
    UNDEFINED,
    "Anim",
    "Inan",
)

CASE2ID, ID2CASE = get_id_mappings(
    UNDEFINED,
    "Acc",
    "Acc2",   # not in GramEval2020
    "Dat",
    "Gen",
    "Gen2",  # !! Par in GramEval2020
    "Ins",
    "Loc",
    "Loc2",   # not in GramEval2020
    "Nom",
    "Voc",
)

GENDER2ID, ID2GENDER = get_id_mappings(
    UNDEFINED,
    "Fem",
    "Masc",
    "Neut",
)

NUMBER2ID, ID2NUMBER = get_id_mappings(
    UNDEFINED,
    # "Count",   # not in GramEval2020; underrepresented, never predicted
    # "Dual",   # not in GramEval2020; underrepresented, never predicted
    "Plur",
    "Sing",
)

# Whole category not in GramEval2020
NAMETYPE2ID, ID2NAMETYPE = get_id_mappings(
    UNDEFINED,
    "Com",
    # "Evn",  # underrepresented, never predicted
    "Geo",
    "Giv",
    # "Oth",  # underrepresented, never predicted
    "Pat",
    "Pro",
    "Prs",
    "Sur",
    # "Zoo",  # underrepresented, never predicted
)

ASPECT2ID, ID2ASPECT = get_id_mappings(
    UNDEFINED,
    "Imp",
    "Perf",
)

MOOD2ID, ID2MOOD = get_id_mappings(
    UNDEFINED,
    "Cnd",
    "Imp",
    # "Imp2",   # not in GramEval2020; underrepresented, never predicted
    "Ind",
)

TENSE2ID, ID2TENSE = get_id_mappings(
    UNDEFINED,
    # "Aor",  # not in GramEval2020; underrepresented, never predicted
    "Fut",
    # "Imp",   # not in GramEval2020; underrepresented, never predicted
    "Past",
    "Pres",
)

# Whole category not in GramEval2020
TRANSIT2ID, ID2TRANSIT = get_id_mappings(
    UNDEFINED,
    "Intr",
    "Intr,Tran",
    "Tran",
)

VERBFORM2ID, ID2VERBFORM = get_id_mappings(
    UNDEFINED,
    "Conv",
    "Fin",
    "Inf",
    "Part",
)

VOICE2ID, ID2VOICE = get_id_mappings(
    UNDEFINED,
    "Act",
    "Mid",
    "Pass",
)

DEGREE2ID, ID2DEGREE = get_id_mappings(
    UNDEFINED,
    "Cmp",
    "Cmp2",  # not in GramEval2020; underrepresented, predicted kinda poorly? sometimes hits though
    "Pos",
    "Sup",
)

PERSON2ID, ID2PERSON = get_id_mappings(
    UNDEFINED,
    "1",
    "2",
    "3",
)

# Whole category not in GramEval2020
NUMFORM2ID, ID2NUMFORM = get_id_mappings(
    UNDEFINED,
    "Combi",
    "Digit",
    "Roman",  # underrepresented, predicted well
    "Word",
)

# Whole category not in GramEval2020
NUMTYPE2ID, ID2NUMTYPE = get_id_mappings(
    UNDEFINED,
    "Card",
    "Frac",
    "Ord",
    "Sets",  # underrepresented, predicted well
)

VARIANT2ID, ID2VARIANT = get_id_mappings(
    UNDEFINED,
    "Short",
)

# Whole category not in GramEval2020
PRONTYPE2ID, ID2PRONTYPE = get_id_mappings(
    UNDEFINED,
    "Dem",
    "Emp",
    # "Exc",  # not in GramEval2020; underrepresented, never predicted
    "Ind",
    "Int",
    "Neg",
    "Prs",
    "Rcp",  # underrepresented, predicted well
    "Rel",
    "Tot",
)

ABBR2ID, ID2ABBR = get_id_mappings(
    UNDEFINED,
    "Yes",
)

# Whole category not in GramEval2020
POSS2ID, ID2POSS = get_id_mappings(
    UNDEFINED,
    "Yes",
)

# Whole category not in GramEval2020
INFLCLASS2ID, ID2INFLCLASS = get_id_mappings(
    UNDEFINED,
    "Ind",  # underrepresented, predicted kinda randomly? sometimes hits though
)

# Whole category not in GramEval2020
REFLEX2ID, ID2REFLEX = get_id_mappings(
    UNDEFINED,
    "Yes",
)

POLARITY2ID, ID2POLARITY = get_id_mappings(
    UNDEFINED,
    "Neg",
)

FOREIGN2ID, ID2FOREIGN = get_id_mappings(
    UNDEFINED,
    "Yes",
)

# Whole category not in GramEval2020
HYPH2ID, ID2HYPH = get_id_mappings(
    UNDEFINED,
    "Yes",  # underrepresented, predicted poorly
)

name2mapping_to_id = {
    "upos": UPOS2ID,
    "Animacy": ANIMACY2ID,
    "Case": CASE2ID,
    "Gender": GENDER2ID,
    "Number": NUMBER2ID,
    "NameType": NAMETYPE2ID,
    "Aspect": ASPECT2ID,
    "Mood": MOOD2ID,
    "Tense": TENSE2ID,
    "Transit": TRANSIT2ID,
    "VerbForm": VERBFORM2ID,
    "Voice": VOICE2ID,
    "Degree": DEGREE2ID,
    "Person": PERSON2ID,
    "NumForm": NUMFORM2ID,
    "NumType": NUMTYPE2ID,
    "Variant": VARIANT2ID,
    "PronType": PRONTYPE2ID,
    "Abbr": ABBR2ID,
    "Poss": POSS2ID,
    "InflClass": INFLCLASS2ID,
    "Reflex": REFLEX2ID,
    "Polarity": POLARITY2ID,
    "Foreign": FOREIGN2ID,
    "Hyph": HYPH2ID,
}


name2mapping_from_id = {
    "upos": ID2UPOS,
    "Animacy": ID2ANIMACY,
    "Case": ID2CASE,
    "Gender": ID2GENDER,
    "Number": ID2NUMBER,
    "NameType": ID2NAMETYPE,
    "Aspect": ID2ASPECT,
    "Mood": ID2MOOD,
    "Tense": ID2TENSE,
    "Transit": ID2TRANSIT,
    "VerbForm": ID2VERBFORM,
    "Voice": ID2VOICE,
    "Degree": ID2DEGREE,
    "Person": ID2PERSON,
    "NumForm": ID2NUMFORM,
    "NumType": ID2NUMTYPE,
    "Variant": ID2VARIANT,
    "PronType": ID2PRONTYPE,
    "Abbr": ID2ABBR,
    "Poss": ID2POSS,
    "InflClass": ID2INFLCLASS,
    "Reflex": ID2REFLEX,
    "Polarity": ID2POLARITY,
    "Foreign": ID2FOREIGN,
    "Hyph": ID2HYPH,
}


names_order = (
    "upos",
    "Animacy",
    "Case",
    "Gender",
    "Number",
    "NameType",
    "Aspect",
    "Mood",
    "Tense",
    "Transit",
    "VerbForm",
    "Voice",
    "Degree",
    "Person",
    "NumForm",
    "NumType",
    "Variant",
    "PronType",
    "Abbr",
    "Poss",
    "InflClass",
    "Reflex",
    "Polarity",
    "Foreign",
    "Hyph",
)
