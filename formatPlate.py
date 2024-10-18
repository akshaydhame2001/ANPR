"""
General Format Logic:
State Code (Part 1): Two uppercase letters, representing the state (except for diplomatic plates and BH-series).
RTO Number (Part 2): A two-digit number representing the district RTO or last two digits of the year in BH series.
Series/Vehicle Class (Part 3):
For private and commercial: Optional 1-3 letters (except O and I).
For BH-series: Optional 1-2 letters (except O and I).
For armed forces: A letter representing the class of vehicle.
Unique Number (Part 4): A unique number between 1 and 9999 (or 6 digits for armed forces).
Diplomatic Plates: Special letters like 'CD' or 'UN' along with a number indicating the diplomatic mission.
"""
import string

def license_complies_format_general(text, type="private"):
    """
    Check if the license plate text complies with the required Indian format based on the type of vehicle.
    
    Args:
        text (str): License plate text.
        type (str): Type of registration (private, vintage, BH, armed_forces, diplomatic, etc.).
    
    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    
    if type == "private" or type == "commercial":
        # State code: First 2 uppercase letters
        if len(text) < 7 or not text[0:2].isupper() or not text[0:2].isalpha():
            return False
        # RTO number: 2 digits
        if not text[2:4].isdigit():
            return False
        # Series: 1-3 letters, optional
        series_part = text[4:-4]  # Extract series if exists
        if series_part and not all(char in string.ascii_uppercase and char not in ['O', 'I'] for char in series_part):
            return False
        # Unique number: 1 to 4 digits
        number_part = text[-4:]
        if not (number_part.isdigit()):
            return False
        return True
    
    elif type == "vintage":
        # Format: State code (2 uppercase letters), 'VA', Series (2 uppercase letters), number (1-4 digits)
        if len(text) != 10:
            return False
        if not text[0:2].isalpha() or text[2:4] != "VA" or not text[4:6].isalpha() or not text[6:10].isdigit():
            return False
        return True
    
    elif type == "BH":
        # Format: Year (2 digits), 'BH', Unique number (1-4 digits), Series (1-2 uppercase letters)
        if len(text) < 9:
            return False
        if not (text[0:2].isdigit() and text[2:4] == "BH" and text[4:8].isdigit()):
            return False
        series_part = text[8:]
        if series_part and not all(char in string.ascii_uppercase and char not in ['O', 'I'] for char in series_part):
            return False
        return True
    
    elif type == "armed_forces":
        # Format: Arrow, Year (2 digits), Class (1 letter), 6-digit number, Check letter
        if len(text) != 10:
            return False
        if text[0] != '↑' or not text[1:3].isdigit() or not text[3].isalpha() or not text[4:10].isdigit():
            return False
        return True
    
    elif type == "diplomatic":
        # Format: 3 digits, Type (CD, CC, UN), 4-digit number
        if len(text) != 10:
            return False
        if not text[0:3].isdigit() or not text[3:5] in ["CD", "CC", "UN"] or not text[5:10].isdigit():
            return False
        return True

    return False

# Test examples
print(license_complies_format_general("KA01AB1234", "private"))  # Expected True
print(license_complies_format_general("KA01VAAB1234", "vintage"))  # Expected True
print(license_complies_format_general("22BH1234A", "BH"))  # Expected True
print(license_complies_format_general("↑22A012345", "armed_forces"))  # Expected True
print(license_complies_format_general("123CD4567", "diplomatic"))  # Expected True


dict_char_to_int = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'J': '3',
    'A': '4',
    'S': '5',
    'G': '6'
    
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'J',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text or False if the format is incorrect.
    """

    license_plate_ = ''
    
    # Mapping logic
    mapping = {
        0: dict_int_to_char,  # State code characters
        1: dict_int_to_char,  # State code characters
        2: dict_char_to_int,   # RTO number
        3: dict_char_to_int,   # RTO number
    }
    
    # Process first two characters (State Code)
    for j in [0, 1]:
        if text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]  # Map state code
        else:
            license_plate_ += text[j]  # Keep original character

    # Process next two characters (RTO Number)
    for j in [2, 3]:
        if text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]  # Map RTO number
        else:
            license_plate_ += text[j]  # Keep original character

    # Process optional series (if exists)
    series_part = text[4:-4]  # Extract series if exists
    for integer in series_part:
        if integer in dict_int_to_char:  # Check against dict_char_to_int for mapping
            license_plate_ += dict_int_to_char[integer]
        else:
            license_plate_ += integer  # Keep original character

    # Process last four characters (Unique Number)
    number_part = text[-4:]  # Unique number part
    for char in number_part:
        if char in dict_char_to_int:  # Map unique number if applicable
            license_plate_ += dict_char_to_int[char]
        else:
            license_plate_ += char  # Keep original character

    return license_plate_

# Example usage
plate_text = "OA014B1234"  # Example plate number
formatted_plate = format_license(plate_text)
print(f"Formatted License Plate: {formatted_plate}")

