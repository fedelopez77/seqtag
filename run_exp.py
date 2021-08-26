SCRIPT = """#!/bin/bash
#SBATCH --job-name=${py_initial}
#SBATCH --output=/hits/basement/nlp/lopezfo/projects/seqtag/out/job-out/out-%j
#SBATCH --error=/hits/basement/nlp/lopezfo/projects/seqtag/out/job-out/err-%j
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${py_nproc}
#SBATCH --gres=gpu:1
#SBATCH --partition=${py_partition}.p
# "SBATCH" --nodelist=skylake-deep-[01]
# "SBATCH" --gres=gpu:1
# "SBATCH" --partition=skylake-deep.p,pascal-deep.p,pascal-crunch.p
# "SBATCH" --exclude=skylake-deep-[01],pascal-crunch-01
# "SBATCH" --nodelist=pascal-deep-[01] to fix the node that I am requesting
# "SBATCH" --mem=250G real memory required per node WARN: exclusive with mem-per-cpu

RUN=$py_run
BRANCH="dev"
INITIAL="$py_initial"
PREP="$py_prep"
RESULTS_FILE="out/$${PREP}.csv"
BS=$py_batch_size
LR=$py_learning_rate
MGN=$py_maxgradnorm
WD=$py_weightdecay
DO=$py_dropout
SEED=$$RANDOM

if [[ $$(hostname -s) = pascal-* ]] || [[ $$(hostname -s) = skylake-* ]]; then
    module load CUDA/10.1.243-GCC-8.3.0
fi
# if [[ $$(hostname -s) = cascade-* ]]; then
#     module load CUDA/10.1.243-GCC-8.3.0
# fi

. /home/lopezfo/anaconda3/etc/profile.d/conda.sh 
conda deactivate
conda deactivate
conda activate seqtag
cd /hits/basement/nlp/lopezfo/projects/seqtag

# if [ "$py_do_pull" == "1" ]; then
#     git co -- .
#     git pull
#     git co $$BRANCH
#     git pull
# fi

MYPORT=`shuf -i 2049-48000 -n 1`
RUN_ID=r$$INITIAL-lr$$LR-mgr$$MGN-bs$$BS-wd$$WD-do$$DO-$$RUN
python -m torch.distributed.launch --nproc_per_node=${py_nproc} --master_port=$$MYPORT train.py \\
    --n_procs=${py_nproc} \\
    --data=data/$$PREP.pt \\
    --run_id=$$RUN_ID \\
    --optim=AdamW \\
    --weight_decay=$$WD \\
    --learning_rate=$$LR \\
    --max_grad_norm=$$MGN \\
    --batch_size=$$BS \\
    --dropout=$$DO \\
    --val_every=1 \\
    --patience=30 \\
    --epochs=300 \\
    --results_file=$$RESULTS_FILE \\
    --job_id=$$SLURM_JOB_ID \\
    --seed=$$SEED > /hits/basement/nlp/lopezfo/projects/seqtag/out/runs/$${RUN_ID}
"""

from string import Template
import itertools
import subprocess
from datetime import datetime


if __name__ == '__main__':
    template = Template(SCRIPT)

    partition = "skylake-deep"
    nprocs = 1
    # models = ["bert-base-uncased"] #"xlm-mlm-en-2048"
    models = ["roberta-base", "distilbert-base-uncased", "facebook/bart-base", "microsoft/layoutlm-base-uncased"]
    #models = ["facebook/bart-base", "microsoft/layoutlm-base-uncased"]
    batch_sizes = [64, 128, 256]
    learning_rates = [1e-5, 3e-5, 6e-5, 9e-5, 2e-4]
    max_grad_norms = [100]
    weight_decays = [0, 1e-3]
    dropouts = [0.2, 0.4]
    runs = [1]
    timestamp = str(datetime.now().strftime("%Y%m%d%H%M%S"))

    for i, (model, bs, lr, mgn, wd, do, run) in enumerate(itertools.product(
            models, batch_sizes, learning_rates, max_grad_norms, weight_decays, dropouts, runs)):
        do_pull = 1 if i == 0 else 0

        model_name = model.replace("_", "-").replace("/", "-")
        prep = f"atis_{model_name}"
        vars = {"py_prep": prep, "py_batch_size": bs, "py_learning_rate": lr,
                "py_maxgradnorm": mgn, "py_weightdecay": wd, "py_dropout": do, "py_initial": model_name,
                "py_nproc": nprocs, "py_run": run, "py_partition": partition, "py_do_pull": do_pull}
        final_script = template.substitute(vars)

        file_name = "job_script.sh"
        with open(file_name, "w") as f:
            f.write(final_script)

        op_res = subprocess.run(["sbatch", file_name], capture_output=True, check=True)
        print(f"{timestamp} - Job {i} vars: {vars} PID: {op_res.stdout.decode()}")
    print("Done!")
