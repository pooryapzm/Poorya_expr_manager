# info: Job sender for M3/Monarch cluster
# author: Poorya

import glob, os, sys
from random import randint
import copy
import yaml
import argparse

parser = argparse.ArgumentParser(description='Experiments statistics.')
parser.add_argument('--config', '-config', action='store', dest='config', help='Path to config file.',
                    default="configs/job_sender.yml")
parser.add_argument('--job_list', '-job_list', action='store', dest='job_list', help='job list',
                    default="job_list.txt")
parser.add_argument('--out_folder', '-out_folder', action='store', dest='out_folder',
                    help='output parent folder',
                    default="models/")
parser.add_argument('--data_folder', '-data_folder', action='store', dest='data_folder', help='data folder',
                    default="../data/")
parser.add_argument('--onmt_path', '-onmt_path', action='store', dest='onmt_path', help='path to OpenNMT folder',
                    default="../code/mtl-onmt/")
parser.add_argument('--conda_env', '-conda_env', action='store', dest='conda_env', help='Conda environment name',
                    default="gradPytorch")
parser.add_argument('--train_template', '-train_template', action='store', dest='train_template',
                    help='train template file',
                    default="configs/train_template.sh")
parser.add_argument('--test_template', '-test_template', action='store', dest='test_template',
                    help='test template file',
                    default="configs/test_template.sh")
parser.add_argument('--debug_mode', '-debug_mode', action='store_true',
          help="debug_mode")
args = parser.parse_args()

###***********************************************
debug_mode = args.debug_mode
out_folder = args.out_folder + "/"
train_template = args.train_template
test_template = args.test_template
job_list = args.job_list
data_folder = args.data_folder
onmt_path = args.onmt_path
conda_env = args.conda_env
train_script = onmt_path + "train.py"
translate_script = onmt_path + "translate.py"
sent_file = out_folder + "sent_jobs.txt"
bleu_script = onmt_path + "tools/multi-bleu.perl"
bpe_remover_script = onmt_path + "cluster_scripts/bpe_remover.py"
###***********************************************

del_com = "sed -i '1d' " + job_list
aux_task_dataset_dict = {
    '0': '/linguistic/amr/onmt_preprocess/amr',
    '1': '/linguistic/tree/onmt_preprocess/tree',
    '2': '/linguistic/ner/onmt_preprocess/ner',
    '3': '/linguistic/pos/onmt_preprocess/pos',
    #
    '9': '/bilingual/deutsch_full-p0/onmt_preprocess/deutsch',
    '10': '/bilingual/deutsch_full-p1/onmt_preprocess/deutsch',
    '11': '/bilingual/deutsch_full-p2/onmt_preprocess/deutsch',
}

# Read config file
with open(args.config, 'r') as stream:
    try:
        configs = yaml.load(stream)
    except yaml.YAMLError as exc:
        raise AssertionError("Cannot read config file. It should be in YAML format.")


# This function is borrowed from ONMT code
def _check_save_model_path(dir):
    save_model_path = os.path.abspath(dir)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)


def output_folder_creator(train_opts, job_opts, out_folder_desc):
    output_folder = out_folder + ""
    for item in out_folder_desc:
        if item in job_opts:
            output_folder += str(job_opts[item]) + "_"
        else:
            output_folder += str(train_opts[item]) + "_"
    output_folder = output_folder.rstrip("_")
    _check_save_model_path(output_folder)
    job_opts['output_folder'] = output_folder


# Concat training data of all tasks, and call ONMT preprocess.py to create the shared vocabulary file (.pt)
def maybe_create_shared_vocab(train_opts, data):
    encoded_data_folder = os.path.abspath(data_folder + train_opts['encode'])
    shared_folder_name = "shared_vocab/" + train_opts['language'] + "_" + train_opts['size'] + "_" + train_opts['task']
    shared_data_path = os.path.abspath(encoded_data_folder + "/" + shared_folder_name)
    if not os.path.exists(shared_data_path):
        print("Create shared vocabulary for tasks: " + shared_data_path)
        os.makedirs(shared_data_path)
        for part in ["training", "development", "testing"]:
            for side in ["src", "tgt"]:
                concat_file = part + ".bpe.unk.flag.tsv." + side
                concat_command = "cat "
                for folder in data.split(","):
                    concat_command += folder[:folder.find("onmt_preprocess")] + part + ".bpe.unk.flag.tsv." + side + " "
                concat_command += " > " + shared_data_path + "/" + concat_file
                os.system(concat_command)
        onmt_preprocess_command = "python3 " + onmt_path + "/preprocess.py"
        for side in ["src", "tgt"]:
            train_part_path = shared_data_path + "/training.bpe.unk.flag.tsv." + side
            onmt_preprocess_command += " -train_" + side + " " + train_part_path
            dev_part_path = shared_data_path + "/development.bpe.unk.flag.tsv." + side
            onmt_preprocess_command += " -valid_" + side + " " + dev_part_path
        save_data_path = shared_data_path + "/onmt_preprocess/"
        _check_save_model_path(save_data_path)
        onmt_preprocess_command += " -save_data " + save_data_path + "mixed"
        # TODO: check whether it was successful or not?
        os.system(onmt_preprocess_command)
    else:
        print("Found shared vocabulary for tasks: " + shared_data_path)

    return shared_data_path + "/onmt_preprocess/mixed"


def create_file_from_template(template_path, new_file_path, fill_dict):
    with open(template_path) as template_file:
        lines = template_file.readlines()
    for i in range(len(lines)):
        line = lines[i]
        for key, value in fill_dict.items():
            line = line.replace("@@%s@@" % key, value)
        lines[i] = line
    new_file = open(new_file_path, 'w')
    for line in lines:
        new_file.write(line)
    new_file.close()
    return


def create_test_file(train_opts, job_opts):
    encoded_data_folder = os.path.abspath(data_folder + train_opts['encode'])
    lang = train_opts['language']
    if os.getenv('my_info') is not None and os.getenv('my_info') == "monarch":
        job_opts['gpu'] = "monarch"

    fill_dict = {}
    fill_dict['ACCOUNT'] = "" if job_opts['gpu'] == "monarch" else "#SBATCH --account=da33"
    fill_dict['JOB_NAME'] = job_opts['test_job_name']
    fill_dict['JOB_TIME'] = '0-48:00:00'
    fill_dict['JOB_LOG'] = os.path.abspath(job_opts['output_folder'] + "/test_job_log")
    fill_dict['PARTITION'] = configs['gpu_partition_map'][job_opts['gpu']]
    fill_dict['QOS'] = "#SBATCH --gres=gpu:V100:1 \n#SBATCH --qos=rtq \n" if job_opts['gpu'] == "rtqp" else ""
    fill_dict['TEST_SRC'] = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts[
        'size'] + "/testing.bpe.flag.tsv.src"
    fill_dict['TEST_TGT'] = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts[
        'size'] + "/testing.bpe.flag.tsv.tgt"
    fill_dict['TEST_TGT_RAW'] = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts[
        'size'] + "/testing.flag.tsv.tgt"
    fill_dict['DEV_SRC'] = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts[
        'size'] + "/development.bpe.flag.tsv.src"
    fill_dict['DEV_TGT'] = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts[
        'size'] + "/development.bpe.flag.tsv.tgt"
    fill_dict['DEV_TGT_RAW'] = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts[
        'size'] + "/development.flag.tsv.tgt"
    fill_dict['BLEU'] = os.path.abspath(bleu_script)
    fill_dict['BPE_REMOVER'] = os.path.abspath(bpe_remover_script)
    fill_dict['TRANSLATE'] = os.path.abspath(translate_script)

    test_script_path = job_opts['output_folder'] + "/test_script"
    create_file_from_template(test_template, test_script_path, fill_dict)
    return test_script_path


def create_train_files(train_opts, job_opts, configs, test_script_path):
    # Step 1: Create job file
    new_job_name = job_opts['output_folder'] + "/job"
    if job_opts['gpu'] == "rtqp":
        job_opts['job_time'] = "0-48:00:00"
    if os.getenv('my_info') is not None and os.getenv('my_info') == "monarch":
        job_opts['gpu'] = "monarch"

    fill_dict = {}
    fill_dict['ACCOUNT'] = "" if job_opts['gpu'] == "monarch" else "#SBATCH --account=da33"
    fill_dict['JOB_NAME'] = job_opts['job_name']
    fill_dict['JOB_TIME'] = job_opts['job_time']
    fill_dict['JOB_LOG'] = os.path.abspath(job_opts['output_folder'] + "/job_log")
    fill_dict['PARTITION'] = configs['gpu_partition_map'][job_opts['gpu']]
    fill_dict['QOS'] = "#SBATCH --gres=gpu:V100:1 \n#SBATCH --qos=rtq \n" if job_opts['gpu'] == "rtqp" else ""
    fill_dict['CONDA_ENV'] = conda_env
    fill_dict['TRAIN'] = os.path.abspath(train_script)
    fill_dict['CONFIG'] = os.path.abspath(job_opts['output_folder'] + "/train.yml")
    fill_dict['TRAING_LOG'] = os.path.abspath(job_opts['output_folder'] + '/train_log')
    fill_dict['TEST_SCRIPT'] = os.path.abspath(test_script_path)

    create_file_from_template(train_template, new_job_name, fill_dict)

    # Step 2: create config file
    encoded_data_folder = os.path.abspath(data_folder + train_opts['encode'])
    lang = train_opts['language']
    # main_task data
    train_data = encoded_data_folder + "/bilingual/" + lang + "_" + train_opts['size'] + "/onmt_preprocess/" + lang
    # aux_task data
    task = train_opts['task']
    num_tasks = 0
    for index in range(len(task)):
        if task[index] == '1':
            num_tasks += 1
            task_data = encoded_data_folder + aux_task_dataset_dict[str(index)]
            train_data += "," + task_data
    train_opts['data'] = train_data

    if train_opts['mtl_shared_vocab'] == '1' and num_tasks > 0:
        shared_vocab_path = maybe_create_shared_vocab(train_opts, train_data)
        train_opts['mtl_shared_vocab_path'] = shared_vocab_path

    save_model_path = os.path.abspath(job_opts['output_folder'] + "/checkpoints/")
    job_opts["save_model_path"] = save_model_path
    _check_save_model_path(save_model_path)

    train_opts['save_model'] = save_model_path + "/model"
    train_opts['analysis_log_file'] = os.path.abspath(job_opts['output_folder']) + "/analysis_log"

    train_opts_cpy = copy.deepcopy(train_opts)
    del train_opts_cpy['task']
    del train_opts_cpy['encode']
    del train_opts_cpy['size']
    del train_opts_cpy['language']
    del train_opts_cpy['mode']

    train_config_path = job_opts['output_folder'] + "/train.yml"
    with open(train_config_path, 'w') as outfile:
        yaml.dump(train_opts_cpy, outfile, default_flow_style=False)

    return new_job_name


def write_log(file_name, message):
    with open(file_name, "a") as log_file:
        log_file.write(message)
        log_file.close()


def merge_two_dicts(dict1, dict2):
    merged_dict = copy.deepcopy(dict1)  # start with dict1's keys and values
    merged_dict.update(dict2)  # modifies merged_dict with dict2's keys and values & returns None
    return merged_dict


def fetch_next_job(job_list):
    while True:
        if os.stat(job_list).st_size == 0:
            return "-1"
        with open(job_list, 'r') as f:
            job = f.readline()
            f.close()
        os.system(del_com)
        if job.strip() == '' or job.startswith("#"):
            continue
        else:
            return job


while True:
    _check_save_model_path(out_folder)

    job = fetch_next_job(job_list)
    if job == "-1":
        exit(0)

    # Extract mode from job description
    mode = ""
    token_job = job.split()
    for token in token_job:
        token_key = token.split("=", 1)[0].strip()
        token_value = token.split("=", 1)[1].strip()
        if token_key == "mode":
            mode = token_value
    mode_default_opts = "%s_default_opts" % mode
    if configs["%s_default_opts" % mode] is None:
        raise AssertionError("default options (%s) has not found in the config file." % mode_default_opts)

    default_job_opts = 'default_opts'
    train_opts = merge_two_dicts(configs[default_job_opts], configs[mode_default_opts])
    job_opts = copy.deepcopy(configs['job_opts'])

    job_seed = randint(0, 9999999)
    job_opts['job_seed'] = str(job_seed)
    job_opts['job_desc'] = '_'.join(job.strip().split())
    token_job = job.split()

    # Fill the job_opts dict with the job description
    for token in token_job:
        token_key = token.split("=", 1)[0].strip()
        token_value = token.split("=", 1)[1].strip()
        if token_key in train_opts:
            train_opts[token_key] = token_value
        elif token_key in job_opts:
            job_opts[token_key] = token_value
        else:
            raise AssertionError("Following option is not supported: " + token_key + "\n" + job)

    job_opts['job_name'] = "m" + str(job_seed)
    job_opts['test_job_name'] = "t" + str(job_seed)
    output_folder_creator(train_opts, job_opts, configs['output_folder_description'])

    # if seed is not determined, use the job_seed as the training seed
    if train_opts['seed'] == "-1":
        train_opts['seed'] = str(job_seed)

    # Create test and train files
    test_script_path = create_test_file(train_opts, job_opts)
    train_script_path = create_train_files(train_opts, job_opts, configs, test_script_path)

    # Create info file
    info_path = job_opts['output_folder'] + "/info"
    with open(info_path, 'w') as outfile:
        yaml.dump(train_opts, outfile, default_flow_style=False)
        yaml.dump(job_opts, outfile, default_flow_style=False)

    # Write job logs
    write_log(sent_file, str(job_seed) + " ||| " + job.strip() + "\n\n")

    command = "sbatch " + train_script_path
    print(command + " ||| GPU: %s" % job_opts['gpu'])
    if not debug_mode:
        os.system(command)
