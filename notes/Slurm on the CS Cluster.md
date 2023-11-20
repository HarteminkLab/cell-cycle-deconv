# Slurm on the CS Cluster
#duke/research

This document will serve as a guide for submitting Python jobs to the cluster. This is what works for me, but there are probably more efficient / better ways to do certain things.

## The Duke CS Server
Information on the slurm batch system can be read here. My instructions are loosely based on what I've read in this page and the wiki it references.
[Slurm Batch System | Department of Computer Science](https://cs.duke.edu/csl/faqs/slurm)

1. Log on to the Computer Science  cluster batch node
```bash
ssh <your_net_id>@login.cs.duke.edu
```

2. Create a directory in xtmp, this is a temporary storage place that is deleted after 30 days. This is where we can pull the repository down and run scripts.
```bash
mkdir /usr/xtmp/<your_net_id>
cd /usr/xtmp/<your_net_id>
```

## Install and initialize Anaconda3 (miniconda3)
We will be using conda environments, so the instructions I have are to install via miniconda—minified version of anaconda for command-line usage on linux.

Download and install miniconda3 into your xtmp directory (or wherever somewhere more permanent if you wish)
```bash
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
```

Clean up/remove the installation script
```bash
rm -rf ./miniconda3/miniconda.sh
```

Initialize conda so it is now available in our shell
```bash
./miniconda3/bin/conda init bash
```

Create a conda environment, in this case we will use an existing conda yaml file that has all of the necessary libraries installed
```bash
conda create -f conda-env.yml
```

If we don't have a yaml file, we can create an empty one, named
```bash
conda create --name <env_name>
```

Then activate it:
```bash
conda activate <env_name>
```

## Interactive shell on cluster node

We can now use that activated environment on the sbatch server. But it will be better if do any computationally rigorous scripting on a cluster node. For now we can run an interactive node, to test any thing we will want to put into a script.

To do that, we will use srun. Which kicks off a slurm job, and give it the user account flag and the flags to indicate we want an interactive bash shell.
```bash
srun --account=<netid> --pty bash
```

The shell should now have a new hostname, indicating that we are now on the new cluster node. Such as:
```bash
<netid>@linux57$ 
```

Notice, anaconda is not available to us. So we will need to run the command to activate access to anaconda and activate our environment.
```bash
. "<path_to_conda>/etc/profile.d/conda.sh"
```

Then activate our environment:
```bash
conda activate <env_name>
```

## Submitting a job to the cluster

Here, we output logs to a directory called slurm-logs (may need to be created), give it a job name, and a path to the script we want to run:
```
sbatch -D ./slurm-logs/ --job-name="my-job" scripts/myscript.sh
```

We can check our running job status with `squeue`:
```bash
squeue -u <netid>
```

An example of `myscript.sh` may look like this. In which we change to the project directory, initialize conda, activate our conda environment, then run our python script. And add any intermediate logging. **Note, any logging in Python will require a `sys.stdout.flush()` flush if we want to check the slurm-logs in real time.**
```bash
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 200G
#SBATCH -p compsci

cd <project-directory>

echo "batch: Starting job on $(date)"

# activate conda and  environment
. "<path_to_anaconda>/etc/profile.d/conda.sh"

# activate environment
conda activate <conda_env>

python mypython.py

echo $(date)
echo "batch: Completed job on $(date)"
```
