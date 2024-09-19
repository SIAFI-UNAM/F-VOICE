""" from https://github.com/keithito/tacotron """

import re
#Valid symbols for spanish language phonemes
valid_symbols = [
    "a", "e", "i", "o", "u", "b", "d̪", "c", "f",
    "g", "k", "l", "m", "n", "ɲ", "ŋ", "p", "r",
    "ɾ", "s", "t̪", "x", "w", "j", "z", "tʃ", "ɡ",
    "dʒ", "ʎ", "β", "ð", "ɣ", "ç", "ɟ", "ɟʝ", "ʃ",
    "ʝ", "θ"
]

#Created the set of symbols
_valid_symbol_set = set(valid_symbols)


class Spanish_CMUDict:
    """Creates a dictionary mapping from a Spanish word to its ARPAbet pronunciations."""

    def __init__(self, file_or_path=None, keep_ambiguous=True):
        # Load the CMUDict dictionary from file.
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding="utf-8") as f:
                #Convert into a python dictionary
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            #Remove the ambiguous pronunciations and keep only one pronunciation
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Returns list of ARPAbet pronunciations of the given word."""
        return self._entries.get(word.upper())

# Regular expression matching an apostrophe followed by a number (e.g. '10')
_alt_re = re.compile(r"\([0-9]+\)")


def _parse_cmudict(file):
    """Parse the CMUDict dictionary file."""
    cmudict = {}
    for line in file:
        if len(line.strip()) == 0 or not line[0].isalpha():
             continue
        parts = line.strip().split('\t')
        #First column have the word
        word = re.sub(_alt_re, "", parts[0])
        #Last column has the pronunciation
        pronunciation = _get_pronunciation(parts[-1])
        if pronunciation:
            if word in cmudict:
                cmudict[word].append(pronunciation)
            else:
                cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    """Convert ARPAbet to CMU pronunciation."""
    parts = s.strip().split(" ")
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return " ".join(parts)
