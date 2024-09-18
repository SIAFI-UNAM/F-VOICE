import re
from unidecode import unidecode
from numbers import normalize_numbers

# Regular expression for whitespace
_whitespace_re = re.compile(r'\s+')

# List of abbreviations and their expansions in Spanish
_abbreviations = [(re.compile(r'\b%s\.' % x[0], re.IGNORECASE), x[1]) for x in [
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
    text = convert_to_ascii(text)
    
    # Collapse multiple whitespace characters into a single space
    text = collapse_whitespace(text)
    
    return text

