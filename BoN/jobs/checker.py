import re

from pathlib import Path

def main():

    job_dir = Path('./code')

    print(job_dir.stem)

    outfiles = [x for x in job_dir.iterdir() if x.suffix == '.out']

    print(len(outfiles))
    # print(outfiles)

    stopped_hpsv2 = []
    stopped_aes = []

    pending = {
        'facedetector': [],
        'styletransfer': [],
        'hpsv2': [],
        'aesthetic': [],
    }

    overall = {
        'facedetector': [],
        'styletransfer': [],
        'hpsv2': [],
        'aesthetic': [],
    }

    for ofile in outfiles:

        if 'b1_' in ofile.stem:
            continue

        overall[ofile.stem.split('_')[-1]].append(ofile.stem)

        lines = []

        with open(ofile, 'r') as fp:
            for line in fp:
                lines.append(line)

        for line in reversed(lines):

            if re.search("CANCELLED AT ", line):

                # if 'aesthetic' in ofile.stem:
                #     stopped_aes.append(ofile.stem)
                # else:
                pending[ofile.stem.split('_')[-1]].append(ofile.stem)
                break

    # print(len(stopped_hpsv2))
    # print(stopped_hpsv2)

    # for key in pending.keys():
    #     print(f'{key}: {1.0 - len(pending[key])/len(overall[key])}')

    print(pending)

if __name__ == '__main__':
    main()