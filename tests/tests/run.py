from tests.tests.model.test_dataset import main as model_dataset_tests
from tests.tests.model.test_model import main as model_creation_tests

# call all python test files from here
# 
# this script, like all others in this project, must be run from the root (the /<repo_name> directory)
# so, the run command should be `python -m tests.tests.run`

tests = [
    model_dataset_tests,
    model_creation_tests
]

test_descriptions = [
    "model/data/dataset.py",
    "model/model.py"
]

tp = 0
tf = 0

stp = 0
stf = 0

for sid in range(1, len(tests)+1):
    print(f"\x1b[36mTest {sid} of {len(tests)}: {test_descriptions[sid-1]}...\x1b[0m")
    
    p, f = tests[sid-1](sid)

    stp += p
    stf += f

    if f == 0:
        print(f'Test {sid} result: \x1b[032mpassed!\x1b[0m')
        tp += 1
    else:
        print(f'Test {sid} result: \x1b[031mfailed! \x1b[0m({p}/{p+f})')
        tf += 1

print(f"--------\nTest Summary:\nPassed {tp} of {tp + tf} tests ({stp} / {stp+stf} subtests)")