""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
from num2words import num2words

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


_spanish_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("sr", "señor"),
        ("sra", "señora"),
        ("srta", "señorita"),
        ("dr", "doctor"),
        ("dra", "doctora"),
        ("prof", "profesor"),
        ("profa", "profesora"),
        ("ing", "ingeniero"),
        ("lic", "licenciado"),
        ("lcda", "licenciada"),
        ("arq", "arquitecto"),
        ("dir", "director"),
        ("gral", "general"),
        ("cap", "capitán"),
        ("ten", "teniente"),
        ("sarg", "sargento"),
        ("cnel", "coronel"),
        ("cd", "ciudad"),
        ("av", "avenida"),
        ("ed", "edificio"),
        ("pág", "página"),
        ("tel", "teléfono"),
        ("dpto", "departamento"),
        ("uds", "ustedes"),
        ("etc", "etcétera"),
        ("ej", "ejemplo"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_spanish_abbreviations(text):
    for regex, replacement in _spanish_abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    """Convierte números en palabras en español usando `num2words`."""
    def replace_number(match):
        num = int(match.group(0))  # Extrae el número
        return num2words(num, lang="es")  # Convierte a palabras en español

    text = re.sub(r'\b\d+\b', replace_number, text)  # Reemplaza números enteros
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def spanish_cleaners(text):
    """Pipeline mejorado para español: convierte a minúsculas, expande números y abreviaturas, y colapsa espacios."""
    text = lowercase(text)  # Convierte a minúsculas
    text = text.replace("»", "")
    text = text.replace("«", "")
    text = expand_numbers(text)  # Convierte números en palabras
    text = expand_spanish_abbreviations(text)  # Expande abreviaturas
    text = collapse_whitespace(text)  # Limpia espacios
    phonemes = phonemize(
        text,
        language="es",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    text = collapse_whitespace(phonemes)  # Limpia espacios
    return phonemes

def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners3(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = backend.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes
