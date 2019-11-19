# Poorya's Experiment Manager (PEM)

A powerful experiment manager for managing (deep learning) experiments

## Using the code

This code is originally developed to work with my Multi-Task Learning models e.g., https://github.com/pooryapzm/AIW_MTL.

To run experiments, you need to put the high-level job descriptions in the job_list.txt file and execute the following command:
```
python job_sender.py -out_folder experiment_folder
```
The script will create a folder for each experiments and generate SLURM scripts for training and testing the model.

To report the status of the models, you can run the following script:

```
python exp_stats.py -folders experiment_folder1,experiment_folder2 
```
This scripts will search for all of the experiments sub-folders in experiment_folder1 and experiment_folder2 folders, and report the results in the following format: 
<center style="padding: 40px"><img width="70%" src="https://pooryapzm.github.io/img/em_sample.png" /></center>

It also creates an excel file (results.xlsx) that contains the value of models' hyper-parameters and results.

## TODO
Add a detailed explanation for the following features:
* The PEM makes experiments and their results searchable by dedicating a unique ID (seed) to each experiments (add an example).
* Efficiently find and fetch experiment results/logs/params by using the experiment ID (add example). 
* ...
