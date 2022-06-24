import signal
import subprocess
from threading import Lock

import background
import sys


# https://gist.github.com/Querela/77b7506b2ce735416e4d77c3bfe954df
# https://github.com/ParthS007/background
###############################################################################
# shell command execution

def run_command(command):
    # run command (can be an array (for parameters))
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # capture output and error
    (output, err) = p.communicate()
    # wait for command to end
    # TODO: really long running?
    status = p.wait()

    # decode output from byte string
    if output is not None:
        output = output.decode('utf-8')
    if err is not None:
        err = err.decode('utf-8')
    # return stdout, stderr, status code
    return (output, err, status)


@background.task
def execute(command):
    debug('run command:', command)
    out, err, status = run_command(command)
    debug('-> exit status:', status)
    debug('output:', out)
    debug('error:', err)

    return (out, err, status)


def execute_now(command):
    debug('run command:', command)
    out, err, status = run_command(command)
    debug('-> exit status:', status)
    debug('output:', out)
    debug('error:', err)

    return (out, err, status)


###############################################################################
# logging

log_lock = Lock()


def log(level, msg, *args, **kwargs):
    log_lock.acquire(True)
    print('[' + level + ']', msg, end=' ')
    for arg in args:
        print(arg, end=' ')
    for k, w in kwargs:
        print('[', k, '=', w, ']', end=' ')
    print()
    log_lock.release()


def info(msg, *args, **kwargs):
    log('INFO   ', msg, *args, **kwargs)


def verbose(msg, *args, **kwargs):
    log('VERBOSE', msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    log('DEBUG  ', msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    log('ERROR  ', msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    log('WARNING', msg, *args, **kwargs)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def wait_for_singal():
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')
    signal.pause()
