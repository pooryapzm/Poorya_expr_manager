import sys

sys.path.append('../')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import RemoteFetcher
from exp_stats import extract_info, train_stats
from utils import setup_logger
import os
import collections

out_file = "outputs/pplx_plot.pdf"


seed_dict = {
    "9723842": ("MT only", "-", False),
    "5358098": ("MTL", "Adaptive scheduler", False),
}

info_list = ["mtl_schedule" ]
rf = RemoteFetcher("m3")
train_log_name = "train_log"
info_log_name = "info"

stats_dict={}
logger = setup_logger()

for index, seed in enumerate(seed_dict.keys()):
    force_fetch = seed_dict[seed][2]
    transfered_files, save_model_path, error_in_connection = \
        rf.fetch_expr_file(seed, [train_log_name, info_log_name], force_fetch=force_fetch, keep_connection=True)

    if error_in_connection:
        logger.info("error in fetching %s"% seed)
        exit(0)

    train_log_file_path = save_model_path + "/" + train_log_name
    info_log_file_path = save_model_path + "/" + info_log_name

    info_dict = collections.OrderedDict()
    for item in info_list:
        info_dict[item] = "Not found!"


    filled_info_dict = extract_info(info_dict, info_log_file_path, train_log_file_path)
    min_pplx, best_epoch, last_epoch, pplx_hist, last_step = train_stats(train_log_file_path)
    filled_info_dict["pplx_hist"] = pplx_hist
    filled_info_dict["seed"] = seed
    filled_info_dict["bleu"] = seed_dict[seed][1]
    filled_info_dict["name"] = seed_dict[seed][0]

    stats_dict['E%s'%index]=filled_info_dict


figs = plt.figure(figsize=(8, 9))
with sns.axes_style("whitegrid"):
    ax1 = figs.add_subplot(211)
    ax2 = figs.add_subplot(212)
    sns.set(style="whitegrid")
    for key in stats_dict:
        exp = stats_dict[key]
        # exp_name = "%s-%s (%s)" % (exp["name"],exp["bleu"], exp['seed'])
        exp_name = exp["name"]
        exp_pplx = exp['pplx_hist']
        exp_stats = [(i + 1, exp_pplx[i]) for i in range(len(exp_pplx))]
        pplx_df = pd.DataFrame(exp_stats, columns=['epoch', 'pplx'])
        ax = sns.lineplot(x="epoch", y="pplx", data=pplx_df, label=exp_name, ax=ax2)
        ax = sns.lineplot(x="epoch", y="pplx", data=pplx_df, label=exp_name, ax=ax1)
    plt.title("Zoomed")
    ax2.set(xlabel='Epoch', ylabel='PPLX')
    ax1.set(xlabel='Epoch', ylabel='PPLX')
    ax2.set(ylim=(5.2, 7))
    ax2.set(xlim=(0, 39))
    ax1.set(ylim=(0, 50))
    plt.tight_layout()

    plt.subplots_adjust(hspace=0.2)

plt.savefig(out_file)
logger.info("PDF file has been saved to: %s" % out_file)
os.system("open %s" % out_file)

exit(0)
