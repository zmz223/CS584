#!/bin/bash
#


### (Recommended)
### Name the project in the batch job queue
#SBATCH -J final_project

### (Optional)
### If you'd like to give a bit more information about your job, you can
### use the command below.
#SBATCH --comment='final project for cs584'

### (REQUIRED)
### Select the queue (also called "partition") to use. The available partitions for your
### use are visible using the 'sinfo' command.
### You must specify 'gpu' or another partition to have access to the system GPUs.
#SBATCH -p gpu

### (REQUIRED for GPU, otherwise do not specify)
### If you select a GPU queue, you must also use the command below to select the number of GPUs
### to use. Note that you're limited to 1 GPU per job as a maximum on the basic GPU queue.
### If you need to use more than 1, contact bmi-it@emory.edu to schedule a multi-gpu test for
### access to the multi-gpu queue.
###
### If you need a specific type of GPU, you can prefix the number with the GPU's type like
### so: "SBATCH -G quadro:1". The available types of GPUs as of 10/31/2019 are:
### turing (12 total)
### titan (only 1; requesting this GPU may result in a delay in your job starting)
### pascal (4 total; using this GPU requires that your code handle being pre-empted/stopped at any
###        time, as there are certain users with priority access to these GPUs).
### volta (8 total) - Coming soon (ie. not available yet; don't request one).
#SBATCH -p rtx -G 1

### (REQUIRED) if you don't want your job to end after 8 hours!
### If you know your job needs to run for a long time or will finish up relatively
### quickly then set the command below to specify how long your job should take to run.
### This may allow it to start running sooner if the cluster is under heavy load.
### Your job will be held to the value you specify, which means that it will be ended
### if it should go over the limit. If you're unsure of how long your job will take to run, it's
### better to err on the longer side as jobs can always finish earlier, but they can't extend their
### requested time limit to run longer.
###
### The format can be "minutes", "hours:minutes:seconds", "days-hours", or "days-hours:minutes:seconds".
### By default, jobs will run for 8 hours if this isn't specified.
#SBATCH -t 3-23:0:0


### (optional) Output and error file definitions. To have all output in a file named
### "slurm-<jobID>.out" just remove the two SBATCH commands below. Specifying the -e parameter
### will split the stdout and stderr output into different files.
### The %A is replaced with the job's ID.
#SBATCH -o %A.out
#SBATCH -e %A.err

### You can specify the number of nodes, number of cpus/threads, and amount of memory per node
### you need for your job. We recommend specifying only memory unless you know you need a
### specific number of nodes/threads, as you will be automatically allocated a reasonable
### amount of threads based on the memory amount requested.

### (REQUIRED)
### Request 4 GB of RAM - You should always specify some value for this option, otherwise
###                       your job's available memory will be limited to a default value
###                       which may not be high enough for your code to run successfully.
###                       This value is for the amount of RAM per computational node.
#SBATCH --mem 40G

### (optional)
### Request 4 cpus/threads - Specify a value for this function if you know your code uses
###                          multiple CPU threads when running and need to override the
###                          automatic allocation of threads based on your memory request
###                          above. Note that this value is for the TOTAL number of threads
###                          available to the job, NOT threads per computational node! Also note
###                          that Matlab is limited to using up to 15 threads per node due to
###                          licensing restrictions imposed by the Matlab software.
##SBATCH -n 4

### (optional)
### Request 2 cpus/threads per task - This differs from the "-n" parameter above in that it
###                                   specifies how many threads should be allocated per job
###                                   task. By default, this is 1 thread per task. Set this
###                                   parameter if you need to dedicate multiple threads to a
###                                   single program/application rather than running multiple
###                                   separate applications which require a single thread each.
##SBATCH -c 2

### (very optional)
### Request 1 node - Only specify a value for this option when you know that your code
###                  will run across multiple systems concurrently. Otherwise you're just
###                  wasting resources that could

### (optional)
### This is to send you emails of job status
### See the manpage for sbatch for all the available options.
#SBATCH --mail-user=zzaiman@emory.edu
#SBATCH --mail-type=ALL

### Your actual commands go below this line. They will be pretty much the same as
### those from the older PBS files.

## sbatch slurm_example_script.sh     #if not using Python



scl enable rh-python36 'python3 train.py'


