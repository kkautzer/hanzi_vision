from tests.tests.utils import run_subset_tests
from model.model import ChineseCharacterCNN

def t_invalid_arch():
    try:
        model = ChineseCharacterCNN(architecture='resnet')
        return False
    except ValueError:
        return True # expected behavior

def t_negative_chars():
    try:
        model = ChineseCharacterCNN(num_classes=-1)
        return False
    except ValueError:
        return True # expected behavior

def t_zero_chars():
    try:
        model = ChineseCharacterCNN(num_classes=0)
        return False
    except ValueError:
        return True # expected behavior    

def t_one_char():
    model = ChineseCharacterCNN(num_classes=1)
    shape = model.googlenet.fc.out_features # check shape of last layer of model
    return shape == 1

def t_1000_chars():
    model = ChineseCharacterCNN(num_classes=1000)
    shape = model.googlenet.fc.out_features # check shape of last layer of model
    return shape == 1000


def main(sid):
    '''
    tests to run for the /model/model.py. Returns a tuple of the number of (passed, failed) tests
    '''

  
    tests = [
        t_invalid_arch,
        t_negative_chars,
        t_zero_chars,
        t_one_char,
        t_1000_chars
    ]

    test_descriptions = [
        "invalid architecture name",
        "negative classes",
        "zero classes",
        "one class",
        "1000 classes"
    ]

    return run_subset_tests(sid, tests, test_descriptions)

if __name__ == "__main__":
    
    main(1)