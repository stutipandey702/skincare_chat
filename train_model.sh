#!/bin/sh

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# 1. setup anaconda environment
# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=skin_env
#research
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

echo "Setting up python environment"
# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
. $ENVDIR/bin/activate

sleep 5


# launch code
echo "Running main.py"
python3 finetune.py "$@"