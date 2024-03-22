def convert_k_m_to_numeric(value):
    """
    Convert values with 'K' or 'M' suffix to float numbers.
    Args:
    - value: The string or numeric value to convert.
    
    Returns:
    - The converted value as float if 'K' or 'M' was found; otherwise, the original value.
    """
    if isinstance(value, str):  # Only process strings
        if value.endswith('K'):
            return float(value[:-1]) * 1e3
        elif value.endswith('M'):
            return float(value[:-1]) * 1e6
    return value