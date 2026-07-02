import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from main import load_data, check_missing_values, check_duplicates, check_value_range

def test_load_data():
    df = load_data()
    assert not df.empty, "DataFrame should not be empty"

def test_check_missing_values():
    df = load_data()
    missing = check_missing_values(df)
    print(f"Columns with missing values:\n{missing}")
    assert "Arrival Delay" in missing.index, "Expected missing values in Arrival Delay"

def test_check_duplicates():
    df = load_data()
    dup_count = check_duplicates(df)
    print(f"Duplicate rows found: {dup_count}")
    assert dup_count >= 0

def test_check_value_range():
    df = load_data()
    invalid = check_value_range(df, "Cleanliness", 1, 5)
    print(f"Invalid Cleanliness ratings: {len(invalid)}")
    assert len(invalid) == 0, "Cleanliness ratings should be within 1-5"
