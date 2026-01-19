from model.data.dataset import get_dataloaders
from tests.tests.utils import run_subset_tests

def main(sid):
    '''
    tests to run for the /model/data/dataset.py file. Returns a tuple of the number of (passed, failed) tests
    '''

    tests = [
        lambda: get_dataloaders(data_dir='./fake/dir'),                 # invalid data_dir
        lambda: get_dataloaders(data_dir='./tests/data/top-5'),                # valid data_dir, 0 workers
        lambda: get_dataloaders(data_dir='./tests/data/top-5', num_workers=1), # valid data_dir, 1 worker
        lambda: get_dataloaders(data_dir='./tests/data/top-5', num_workers=4)  # valid data_dir, 4 workers
    ]
    test_descriptions = [
        "Invalid data_dir",
        "Valid data_dir, 0 workers",
        "Valid data_dir, 1 worker",
        "Valid data_dir, 4 workers"
    ]

    return run_subset_tests(sid, tests, test_descriptions)

if __name__ == '__main__':
    sid = 1
    print(f"\x1b[36mTest {sid}: dataset.py...\x1b[0m")
    p, f = main(sid)

    if f == 0:
        print(f'Test {sid} result: \x1b[032mpassed!\x1b[0m')
    else:
        print(f'Test {sid} result: \x1b[031mfailed! \x1b[0m({p}/{p+f})')