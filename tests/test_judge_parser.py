import pytest
from src.annotate import safe_parse_json

def test_safe_parse_json_valid():
    assert safe_parse_json('{"unsupported_label": 1}') == {"unsupported_label": 1}
    
def test_safe_parse_json_with_text():
    text = '''Sure! Here is the JSON:
    {
       "utility_label": "useful"
    }
    Hope this helps!'''
    assert safe_parse_json(text) == {"utility_label": "useful"}
    
def test_safe_parse_json_invalid():
    assert safe_parse_json('No json here') is None
