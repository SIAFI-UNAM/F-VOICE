""" from https://github.com/keithito/tacotron """

#import inflect
import re
from num2words import num2words  # Import library to convert numbers to words


#_inflect = inflect.engine() not useful for spanish
# Regular expressions for matching different types of numbers
_COMMA_NUMBER_RE = re.compile(r'([0-9][0-9\,]+[0-9])')
_DECIMAL_NUMBER_RE = re.compile(r'([0-9]+\.[0-9]+)')
_PESOS_RE = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ORDINAL_RE = re.compile(r'[0-9]+(º|ª|°|er|do|da|ro|ra|to|ta)')
_NUMBER_RE = re.compile(r'[0-9]+')


def _remove_commas(m):
    """
    Remove commas from a number string.

    Args:
        match (re.Match): The regex match object.

    Returns:
        str: The number string without commas.
    """
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    """
    Replace decimal points with ' punto ' in a number string.

    Args:
        match (re.Match): The regex match object.

    Returns:
        str: The number string with ' punto ' replacing the decimal point.
    """
    return m.group(1).replace(".", " punto ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"

def _expand_pesos(match):
    """
    Convert currency values from pesos to words.

    Args:
        match (re.Match): The regex match object containing the currency value.

    Returns:
        str: The currency value converted to words.
    """
    amount = match.group(1)
    parts = amount.split('.')
    
    # Handle unexpected format with more than 2 parts
    if len(parts) > 2:
        return amount + ' pesos'  # Unexpected format

    pesos = int(parts[0]) if parts[0] else 0
    centavos = int(parts[1]) if len(parts) > 1 and parts[1] else 0

    if pesos and centavos:
        peso_unit = 'peso' if pesos == 1 else 'pesos'
        centavo_unit = 'centavo' if centavos == 1 else 'centavos'
        return f'{num2words(pesos, lang="es")} {peso_unit}, {num2words(centavos, lang="es")} {centavo_unit}'
    elif pesos:
        peso_unit = 'peso' if pesos == 1 else 'pesos'
        return f'{num2words(pesos, lang="es")} {peso_unit}'
    elif centavos:
        centavo_unit = 'centavo' if centavos == 1 else 'centavos'
        return f'{num2words(centavos, lang="es")} {centavo_unit}'
    else:
        return 'cero pesos'



# def _expand_ordinal(m):
#     return _inflect.number_to_words(m.group(0))

def _expand_ordinal(match):
    """
    Convert ordinal numbers to words.

    Args:
        match (re.Match): The regex match object containing the ordinal number.

    Returns:
        str: The ordinal number converted to words.
    """
    return num2words(match.group(0), lang='es', to='ordinal')

# def _expand_number(m):
#     num = int(m.group(0))
#     if num > 1000 and num < 3000:
#         if num == 2000:
#             return "two thousand"
#         elif num > 2000 and num < 2010:
#             return "two thousand " + _inflect.number_to_words(num % 100)
#         elif num % 100 == 0:
#             return _inflect.number_to_words(num // 100) + " hundred"
#         else:
#             return _inflect.number_to_words(
#                 num, andword="", zero="oh", group=2
#             ).replace(", ", " ")
#     else:
#         return _inflect.number_to_words(num, andword="")



def _expand_number(match):
    """
    Convert regular numbers to words.

    Args:
        match (re.Match): The regex match object containing the number.

    Returns:
        str: The number converted to words.
    """
    num = int(match.group(0))
    return num2words(num, lang='es')

# def normalize_numbers(text):
#     text = re.sub(_comma_number_re, _remove_commas, text)
#     text = re.sub(_pounds_re, r"\1 pounds", text)
#     text = re.sub(_dollars_re, _expand_dollars, text)
#     text = re.sub(_decimal_number_re, _expand_decimal_point, text)
#     text = re.sub(_ordinal_re, _expand_ordinal, text)
#     text = re.sub(_number_re, _expand_number, text)
#     return text


def normalize_numbers(text):
    """
    Normalize numbers in a text by converting them to words and handling currency.

    Args:
        text (str): The input text containing numbers.

    Returns:
        str: The text with numbers and currency normalized.
    """
    text = re.sub(_COMMA_NUMBER_RE, _remove_commas, text)
    text = re.sub(_PESOS_RE, _expand_pesos, text)
    text = re.sub(_DECIMAL_NUMBER_RE, _expand_decimal_point, text)
    text = re.sub(_ORDINAL_RE, _expand_ordinal, text)
    text = re.sub(_NUMBER_RE, _expand_number, text)
    return text
