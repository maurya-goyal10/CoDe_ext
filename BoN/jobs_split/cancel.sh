# loop over all sbatch files in the directory, print the filename and submit the job to SLURM
#

for i in {11574246..11574261}; do
    scancel ${i}
done