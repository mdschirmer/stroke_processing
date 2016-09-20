#!/usr/bin/env bash

site=$1
subjectlist=$2
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source $DIR/../stroke.cfg 2>/dev/null

export PYTHONPATH=${pipebuilder_path}:$DIR/../:$PYTHONPATH

for subj in $subjectlist; do
    echo $subj
    python $DIR/../stroke_processing/registration/flairpipe.py ${subj} 9.0 0.2 ${processing_root}/data/${site}/
done
