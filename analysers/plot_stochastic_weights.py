import sys

sys.path.append('../')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import RemoteFetcher
from utils import dataframe_to_pdf
from PyPDF2 import PdfFileWriter, PdfFileReader
from exp_stats import extract_info, train_stats
from utils import setup_logger
from utils import fetch_expr_file_server_list
import os
import yaml
import collections

seed = 4284667


pattern = "[TASK_LIST"
skip_but_count_pattern="[TASK_LIST-M"
# Force fetch the log, otherwise it uses the downloaded log
force_fetch = False
# Name of the server or use None to search for all!
server = None
#
skip_ratio = 10
avg_steps = 100
step_limits = [0, 1000000]
deducted_bias = 0
show_epochs = False
# Sample line: [2019-04-23 16:28:07,200 INFO] [META_AIW] Step: 300, Batch: 0 ||| 266.24  266.24  204.8  286.72
key_map = {
    "Step": -1,
    "T0: Translation": 4,
    "T1: Semantic": 5,
    "T2: Syntactic": 6,
    "T3: NER": 7,
}

config_file = "configs/general.yaml"


def _line_to_row(line, line_no):
    if line_no % skip_ratio == 0 and line_no < step_limits[1]:
        row = []
        segmented_line = line.split()
        for key in key_map.keys():
            index = key_map[key]
            if index == -1:
                if key == "Step":
                    row.append(str(line_no))
            else:
                if index < len(segmented_line):
                    row.append(segmented_line[index].strip("[").strip("]").strip(",").strip())
                else:
                    row.append("")
        return row
    else:
        return None


def _show_epochs(show_pplx=False):
    y_label_pos = (max(pplx_hist) + min(pplx_hist)) / 2
    for index, delim in enumerate(current_epochs_delims):
        color = "blue" if index + 1 == best_epoch else "red"
        plt.axvline(x=delim, color=color, ls="dotted")
        if show_pplx:
            plt.text(delim, y_label_pos, round(pplx_hist[index], 2), ha='center', va='center', rotation='vertical')


def pdf_merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()

    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))

    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)


# Read config file
with open(config_file, 'r') as stream:
    try:
        configs = yaml.load(stream)
    except yaml.YAMLError as exc:
        raise AssertionError("Cannot read config file. It should be in YAML format.")

server_list = configs["server_list"]
meta_log_name = configs["meta_log_name"]
train_log_name = configs["train_log_name"]
info_log_name = configs["info_log_name"]


logger = setup_logger()

if server is not None:
    server_list = [server]

save_model_path = fetch_expr_file_server_list(seed, server_list, [meta_log_name, train_log_name, info_log_name],
                                              force_fetch)
if save_model_path == "":
    exit(-1)
else:
    meta_log_file_path = save_model_path + "/" + meta_log_name
    train_log_file_path = save_model_path + "/" + train_log_name
    info_log_file_path = save_model_path + "/" + info_log_name
    out_file = "%s/weight_plot_%s.pdf" % (save_model_path, seed)

info_dict = collections.OrderedDict()
colored_dict = collections.OrderedDict()
for item in configs["info"]:
    if isinstance(item, dict):
        key = list(item.keys())[0]
        color = item[key]
        info_dict[key] = "Not found!"
        colored_dict[key] = color
    else:
        info_dict[item] = "Not found!"

filled_info_dict = extract_info(info_dict, info_log_file_path, train_log_file_path)
min_pplx, best_epoch, last_epoch, pplx_hist, last_step = train_stats(train_log_file_path)

data_frame = pd.DataFrame(columns=list(key_map.keys()))
with open(meta_log_file_path) as log_file:
    counter = -1
    for line in log_file:
        if pattern in line:
            counter += 1
            row = _line_to_row(line, counter)
            if row is not None:
                data_frame.loc[data_frame.shape[0]] = row
        elif skip_but_count_pattern in line:
            counter+=1

for key in key_map.keys():
    data_frame[key] = pd.to_numeric(data_frame[key])
    if key.startswith("T"):
        data_frame[key] = data_frame[key] - deducted_bias

epochs_delims = []

def dataframe_avg(df, avg_steps):
    # df_out = df.groupby(['symbol_a', 'symbol_b', g // 3]).mean().reset_index(level=2, drop=True).reset_index()
    df_out = df.groupby(df.index // avg_steps).mean()
    return df_out


# data_frame = data_frame.iloc[::skip_ratio]
if avg_steps is not None:
    data_frame = dataframe_avg(data_frame, avg_steps)
data_frame = data_frame[step_limits[0] < data_frame['Step']]
data_frame = data_frame[data_frame['Step'] < step_limits[1]]
current_epochs_delims = list(filter(lambda x: step_limits[0] < x < step_limits[1], epochs_delims))
current_pplx_hist = pplx_hist[:len(current_epochs_delims)]
current_epochs_delims = current_epochs_delims[:len(current_pplx_hist)]
# dataframe_to_pdf(data_frame); exit(0)

figs = plt.figure(figsize=(8, 9))
with sns.axes_style("whitegrid"):
    ax1 = figs.add_subplot(211)
    sns.set(style="whitegrid")
    for key in key_map.keys():
        if key.startswith("T"):
            ax = sns.lineplot(x="Step", y=key, data=data_frame, label=key.split(":")[1].strip(), ax=ax1)
            # data_frame[key].max()

    if show_epochs:
        _show_epochs()
    plt.title("Model ID: %s" % str(seed))
    ax1.set(xlabel='Step', ylabel='Weight')
    plt.tight_layout()
    plt.xlim(data_frame["Step"].min(), data_frame["Step"].max())

    plt.subplots_adjust(hspace=0.2)

    # with sns.axes_style("whitegrid"):
    ax2 = figs.add_subplot(212)
    ax = sns.lineplot(x=current_epochs_delims, y=current_pplx_hist, label="PPLX", ax=ax2)
    if show_epochs:
        _show_epochs(show_pplx=True)

    plt.xlim(data_frame["Step"].min(), data_frame["Step"].max())
    # ax2.set_ylim([12, 20])
    plt.subplots_adjust(bottom=0.1)

plt.savefig(out_file + "_fig.pdf")

# Retrive model info and create INFO PDF
info_df = pd.DataFrame(filled_info_dict.items(), columns=['Param', 'Value'])
dataframe_to_pdf(info_df, out_file + "_info")

# Merge Figure and Info PDFs
pdf_merger(out_file, [out_file + "_fig.pdf", out_file + "_info.pdf"])
os.remove(out_file + "_fig.pdf")
os.remove(out_file + "_info.pdf")
os.remove(out_file + "_info.tex")

# _attach_text_pdf(out_file, text_list)

logger.info("PDF file has been saved to: %s" % out_file)
os.system("open %s" % out_file)

exit(0)
