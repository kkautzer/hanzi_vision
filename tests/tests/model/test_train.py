from tests.tests.utils import run_subset_tests


# invalid nchars (negative)

# invalid nchars (zero)

# invalid nchars (too many)

# negative epochs

# resume nonexistent model


# train 5 chars model

# resume an invalid epoch (negative and not -1)

# resume an invalid epoch (too high)

# resume training

# existing [invalid] model name

# second model w/ different number of characters

# third model w/ same number of characters as first

# resume another model (different from first)

def main(sid):
    
    tests = [

    ]
    
    test_descriptions = [
        "nchars - negative",
        "nchars - zero",
        "nchars - exceed limit"
        "negative epochs",
        "resume a nonexistent model",
        "train basic model",
        "invalid resume epoch - negative and not -1",
        "invalid resume epoch - too high",
        "resume model training",
        "new model with existing name",
        "train second model - different nchars",
        "train third model - same nchars as first",
        "resume another model - different from first"
    ]

    return run_subset_tests(tests, test_descriptions)