"""
Answer extraction and comparison helpers

Shared GSM8K answer extraction, normalization, and comparison.
Keeps training and inference on the same evaluation rule.
"""

import re
import ast
import json


def parse_answer_field(answer_field):
    """
    Parse an answer field in several formats:
    - Plain string: "128"
    - List: [128] or ["128"]
    - Stringified list: "['Final Answer: 128']"
    - JSON string: '["128"]'
    
    Returns: a cleaned string
    """
    # 1. If it is already a list, take the first element
    if isinstance(answer_field, list):
        if len(answer_field) > 0:
            answer_field = answer_field[0]
        else:
            return ""
    
    # 2. If it is a string, try to parse it as a list
    if isinstance(answer_field, str):
        # Try JSON
        try:
            parsed = json.loads(answer_field)
            if isinstance(parsed, list) and len(parsed) > 0:
                answer_field = parsed[0]
        except (json.JSONDecodeError, ValueError):
            # Try ast.literal_eval
            try:
                parsed = ast.literal_eval(answer_field)
                if isinstance(parsed, list) and len(parsed) > 0:
                    answer_field = parsed[0]
            except (ValueError, SyntaxError):
                pass  # leave it unchanged
    
    # 3. Convert to a string and clean it
    answer_str = str(answer_field).strip()
    
    # 4. Strip common prefixes
    prefixes = [
        r'^\s*Final Answer\s*:\s*',
        r'^\s*The answer is\s*:\s*',
        r'^\s*Answer\s*:\s*',
        r'^\s*####\s*',  # GSM8K sometimes marks the answer with ####
    ]
    for prefix in prefixes:
        answer_str = re.sub(prefix, '', answer_str, flags=re.IGNORECASE)
    
    return answer_str.strip()


def extract_number_from_text(text):
    """
    Extract the final numeric answer from text (several formats, in priority order)
    
    Formats, in priority order:
    1. <ans>...</ans> tag (scheme 3)
    2. Standard format: <think>...</think>\n42
    3. Final Answer format: <think>...Final Answer: 42</think>
    4. No format: take the last number
    
    Returns: (number_str, number_value)
        - number_str: normalized number string (for example "128")
        - number_value: numeric value (float), or None if none can be extracted
    """
    # 1. Prefer an <ans>...</ans> tag (the scheme-3 format)
    ans_pattern = r'<ans>(.+?)</ans>'
    ans_matches = re.findall(ans_pattern, text, re.IGNORECASE | re.DOTALL)
    if ans_matches:
        text = ans_matches[-1].strip()  # use the last <ans> tag
    
    # 2. Fallback: text after </think> (older format)
    elif "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        
        # If there is real text after </think>, use that
        if after_think:
            text = after_think
        else:
            # If </think> is followed by nothing, try to pull "Final Answer: xxx" from inside
            final_answer_pattern = r'Final Answer[:\s]+(.+?)(?:\n|</think>|$)'
            matches = re.findall(final_answer_pattern, text, re.IGNORECASE)
            if matches:
                text = matches[-1].strip()
    
    # 3. Extract numbers with a regex (negatives, decimals, thousands separators)
    # Pattern:-?[\d,]+\.?\d*
    # Examples:-1,234.56、1234、-5.5、1,000,000
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    
    if not numbers:
        return text.strip(), None
    
    # 4. Take the last number (usually the final answer)
    last_number = numbers[-1]
    
    # 5. Clean and normalize
    # Remove thousands separators
    cleaned = last_number.replace(',', '')
    
    # 6. Convert to a number
    try:
        num_value = float(cleaned)
        
        # 7. Normalize: integers have no decimal point; decimals stay as they are
        if num_value.is_integer():
            return str(int(num_value)), num_value
        else:
            return str(num_value), num_value
    except ValueError:
        return cleaned, None


def extract_answer(text_or_field):
    """
    Shared answer-extraction function
    
    Arguments:
        text_or_field: generated text, a final_answer field, and so on
    
    Returns:
        normalized answer string (one numeric form)
    """
    # 1. Parse the field first (lists, JSON, and similar)
    parsed_text = parse_answer_field(text_or_field)
    
    # 2. extract the number
    answer_str, answer_num = extract_number_from_text(parsed_text)
    
    # 3. Prefer the normalized number string
    return answer_str


def compare_answers(predicted, ground_truth, tolerance=1e-6):
    """
    Compare two answers, preferring a numeric comparison
    
    Arguments:
        predicted: predicted answer (text or number)
        ground_truth: reference answer (text or number)
        tolerance: tolerance for float comparison
    
    Returns:
        bool: whether they match
    """
    # 1. Extract normalized answers
    pred_str = parse_answer_field(predicted)
    gt_str = parse_answer_field(ground_truth)
    
    # 2. Extract numbers
    _, pred_num = extract_number_from_text(pred_str)
    _, gt_num = extract_number_from_text(gt_str)
    
    # 3. Prefer a numeric comparison
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < tolerance
    
    # 4. Fall back to a string comparison (ignore case and whitespace)
    return pred_str.lower().strip() == gt_str.lower().strip()
