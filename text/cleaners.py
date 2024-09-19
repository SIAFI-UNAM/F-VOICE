""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


# Regular expression matching whitespace:
import re
from unidecode import unidecode
from .numbers import normalize_numbers
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations (english or spanish):
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
##spanish
    ('sr', 'señor'),
    ('sra', 'señora'),
    ('dr', 'doctor'),
    ('dra', 'doctora'),
    ('st', 'san'),
    ('srta', 'señorita'),
    ('av', 'avenida'),
    ('pág', 'página'),
    ('vol', 'volumen'),
    ('prof', 'profesor'),
    ('profa', 'profesora'),
    ('ud', 'usted'),
    ('uds', 'ustedes'),
    ('ing', 'ingeniero')
]]


def expand_abbreviations(text):
    """
    Expand abbreviations in the text based on predefined mappings.
    
    Args:
        text (str): The input text with abbreviations.
    
    Returns:
        str: The text with abbreviations expanded.
    """
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    """
    Convert numbers in the text to their word representation.
    
    Args:
        text (str): The input text containing numbers.
    
    Returns:
        str: The text with numbers expanded to words.
    """
    return normalize_numbers(text)


def lowercase(text):
    """
    Convert all characters in the text to lowercase.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The text in lowercase.
    """
    return text.lower()


def collapse_whitespace(text):
    """
    Replace multiple whitespace characters with a single space.
    
    Args:
        text (str): The input text with possible multiple whitespace characters.
    
    Returns:
        str: The text with collapsed whitespace.
    """
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    """
    Convert the text to ASCII, removing accents and special characters.
    
    Args:
        text (str): The input text with possible accents and special characters.
    
    Returns:
        str: The ASCII representation of the text.
    """
    return unidecode(text)


def basic_cleaners(text):
    """
    Basic cleaning pipeline that converts text to lowercase and collapses whitespace.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The cleaned text.
    """
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """
    Pipeline for non-Spanish text that transliterates to ASCII.
    
    Args:
        text (str): The input text with possible non-ASCII characters.
    
    Returns:
        str: The text transliterated to ASCII.
    """
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def spanish_cleaners(text):
    """
    Pipeline for Spanish text, including expansion of abbreviations, conversion to ASCII, expansion of numbers, and collapsing whitespace.
    
    Args:
        text (str): The input Spanish text.
    
    Returns:
        str: The cleaned and expanded text.
    """
    # Convert text to lowercase to ensure uniform processing
    text = lowercase(text)
    
    # Expand abbreviations before converting to ASCII
    text = expand_abbreviations(text)

    # Convert numbers in the text to their word representation
    text = expand_numbers(text)
    
    # Convert text to ASCII to handle special characters and accents
    #text = convert_to_ascii(text)
    #in spanish we don't need to convert to ascii because we need special characters
    
    # Collapse multiple whitespace characters into a single space
    text = collapse_whitespace(text)
    
    return text
