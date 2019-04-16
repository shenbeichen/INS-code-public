This is an example of NaBr at 700K with Ei=30meV.

1. change the paths and filenames of input data and output results if needed.

2. Workflow: candidate_bp ---> bps ---> q-offfset ---> cut_with_mps.

3. I also attached sricpts for folding back the whole dataset, in the folder named 'q-offset-all' and 'cut_with_mps_all'

4. ON NERSC (interactive mode):

salloc -N 2 -t 30 -C haswell -q interactive --image=nersc/spark-2.3.0:v1 --volume="/global/cscratch1/sd/<user_name>/tmpfiles:/tmp:perNodeCache=size=200G"

module load python3

module load spark/2.3.0

start-all.sh

shifter spark-submit --master $SPARKURL <path_to_python_script>

stop-all.sh

exit
