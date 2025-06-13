# loop over all sbatch files in the directory, print the filename and submit the job to SLURM
#

for FILE in *.sbatch; do
    if [[ "${FILE}" == *"code40"* ]]; then
        echo ${FILE}
        sbatch ${FILE}
        sleep 1
    else
        echo "Not submitting template."
    fi
done