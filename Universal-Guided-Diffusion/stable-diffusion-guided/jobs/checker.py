import re

from pathlib import Path

def main():

    job_dir = Path('.')

    outfiles = [x for x in job_dir.iterdir() if x.suffix == '.out']

    print(len(outfiles))

    stopped_hpsv2 = []
    stopped_aes = []

    for ofile in outfiles:

        # if not '100' in ofile.stem:
        #     continue

        lines = []

        with open(ofile, 'r') as fp:
            for line in fp:
                lines.append(line)

        for line in reversed(lines):

            if re.search("CANCELLED AT ", line):

                if 'aes' in ofile.stem:
                    stopped_aes.append(ofile.stem)
                else:
                    stopped_hpsv2.append(ofile.stem)
                break

    print(len(stopped_aes))
    print(stopped_aes)

if __name__ == '__main__':
    main()