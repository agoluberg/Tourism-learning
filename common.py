
"""
common constants and variables
"""

VIRTUAL_ENV_ROOT = '/home/ago/py'
VIRTUAL_ENV_MILK = VIRTUAL_ENV_ROOT + '/milk'
VIRTUAL_ENV_SCIKIT = VIRTUAL_ENV_ROOT + '/scikit-learn-dev'

DATA_FILES_PATH = 'data'
RESULT_FILES_PATH = 'results'
LEARNER_MODULES_PATH = 'learners'

"""
Everything else is put into "files":
models, predictions, helper files, etc
"""
FILES_PATH = 'files'

def default_subprocess_command(module_name, trainfile, testfile='', resultfile='', outputfile=''):
    cmd = 'python run_learner.py -m ' + module_name
    if trainfile:
        cmd += ' -t ' + trainfile
    if testfile:
        cmd += ' -p ' + testfile
    if resultfile:
        cmd += ' -r ' + resultfile
    return cmd
