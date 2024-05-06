
import re
from num2words import num2words
import os


def roman_to_arabic(roman_numeral):
    """Converts from roman to arabic numbers"""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    arabic_numeral = 0
    prev_value = 0

    for char in roman_numeral:
        value = roman_values[char]
        if value > prev_value:
            arabic_numeral += value - 2 * prev_value
        else:
            arabic_numeral += value
        prev_value = value

    return arabic_numeral

def normalize_text(text, language):
    """Normalize text"""
    # Remove all special characters but !¡ and ?¿
    text = re.sub(r"([^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ ,¡!¿?])", "", text)
    # identify and normalize arabic numbers
    if language == 'es':
        text = ' '.join([num2words(word.replace(',', ''), lang='es') if word.replace(',', '').isdigit() else word for word in text.split()])
    else:
        text = ' '.join([num2words(word.replace(',', '')) if word.replace(',', '').isdigit() else word for word in text.split()])

    return text  

def normalize_txt(input_file, output_file, language):
    # Open txt file
    with open(input_file, "r", encoding="latin1") as file:
        lines = file.readlines()

    # List to store the updated lines
    updated_lines = []

    # Process each line
    for line in lines:
        # Split the line into three parts using "|" as separator
        file_name, original_text = line.split("|", 1)
        original_text = original_text.strip()

        # Normalize the original text
        normalized_text = normalize_text(original_text, language)

        # Update the line with the normalized text
        new_line = f"{file_name} | {original_text} | {normalized_text}.\n"
        updated_lines.append(new_line)

    # Write the updated lines to the file
    with open(output_file, "w") as updated_file:
        updated_file.writelines(updated_lines)

    updated_file_path = os.path.abspath(output_file)
    print("Updated file saved as:", updated_file_path)
    return updated_file_path