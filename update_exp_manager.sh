#!/usr/bin/env bash
file_list="configs exp_stats.py job_sender.py update_exp_manager.sh"
for file in ${file_list}; do
    cp -r ../code/mtl-onmt/cluster_scripts/experiment_manager/${file} ./
done