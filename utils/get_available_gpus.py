import sys
import os


def get_gpus_without_triggering_pytorch_cuda_initialization(wd, envvars):
    import subprocess
    # temporary work dir for running benchmarks!!
    out = subprocess.run([sys.executable, os.path.join(wd, 'utils/get_available_gpus.py')],
                         capture_output=True,
                         env=envvars)
    text_output = out.stdout.decode()
    err_output = out.stderr.decode()
    returncode = out.returncode

    from utils.utils import log
    if returncode or err_output:
        log.error(
            'Querying available GPUs... return code %d, error: %s, stdout: %s',
            returncode,
            err_output,
            text_output,
        )

    log.debug('Queried available GPUs: %s', text_output)
    return text_output.replace('\n', '')


def main():
    import torch
    device_count = torch.cuda.device_count()
    available_gpus = ','.join(str(g) for g in range(device_count))
    print(available_gpus)
    return 0


if __name__ == '__main__':
    sys.exit(main())
