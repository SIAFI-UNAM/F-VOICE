import re
from num2words import num2words  # Import library to convert numbers to words

# Regular expressions for matching different types of numbers
_COMMA_NUMBER_RE = re.compile(r'([0-9][0-9\,]+[0-9])')
_DECIMAL_NUMBER_RE = re.compile(r'([0-9]+\.[0-9]+)')
_PESOS_RE = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ORDINAL_RE = re.compile(r'[0-9]+(º|ª|°|er|do|da|ro|ra|to|ta)')
_NUMBER_RE = re.compile(r'[0-9]+')


def _remove_commas(match):
    """
    Remove commas from a number string.

    Args:
        match (re.Match): The regex match object.

    Returns:
        str: The number string without commas.
    """
    return match.group(1).replace(',', '')


def _expand_decimal_point(match):
    """
    Replace decimal points with ' punto ' in a number string.

    Args:
        match (re.Match): The regex match object.

    Returns:
        str: The number string with ' punto ' replacing the decimal point.
    """
    return match.group(1).replace('.', ' punto ')


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


def _expand_ordinal(match):
    """
    Convert ordinal numbers to words.

    Args:
        match (re.Match): The regex match object containing the ordinal number.

    Returns:
        str: The ordinal number converted to words.
    """
    return num2words(match.group(0), lang='es', to='ordinal')


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
