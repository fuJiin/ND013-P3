# Usage: drive.sh <model_filename>
scp carnd@$GPU_BOX:~/$1 ./$1
python drive.py $1
