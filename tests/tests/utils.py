import contextlib
import os
import sys

def run_subset_tests(section_id, tests, test_descriptions):
    '''
    Runs a parameterized set of tests

    :param section_id: Test number identifying the larger test that these subtests are a part of
    :param tests: Tuple/list of test functions
    :param test_descriptions: Tuple/list of test descriptions. Used for output only.

    :return: Tuple containing the number of (passed, failed) tests.
    '''
    passed = 0
    failed = 0

    os.makedirs('./tests/output/out', exist_ok=True)
    os.makedirs('./tests/output/err', exist_ok=True)

    for tid in range(1, len(tests)+1):
        print(f"\tTest {section_id}.{tid} ({test_descriptions[tid-1]})...",end="", flush=True)
        try:
            with open(f'./tests/output/out/{section_id}.{tid}.txt', 'w') as f:
                with contextlib.redirect_stdout(f):
                    with open(f'./tests/output/err/{section_id}.{tid}.txt', 'w') as g:
                        with contextlib.redirect_stderr(g):
                            r = tests[tid-1]()

            if r:
                passed += 1
                print(f'\t\x1b[32mpassed!\x1b[0m')
            else:
                failed += 1
                print(f'\t\x1b[31mfailed!\x1b[0m', flush=True)

            # passed += 1 if r == True else failed += 1

        except Exception as e:
            with open(f'./tests/output/err/{section_id}.{tid}.txt', 'a') as f:
                f.write(str(e))

            failed += 1
            print(f'\t\x1b[31mfailed!\x1b[0m', flush=True)

        finally:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

    return passed, failed