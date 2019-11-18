
from utils import RemoteFetcher

force_fetch = False
seed = 3545897

#
log_name = "info"

rf = RemoteFetcher()
transfered_files, save_model_path, error_in_connection = rf.fetch_expr_file(seed, [log_name], force_fetch=force_fetch)
log_file_path = save_model_path + "/" + log_name

with open(log_file_path) as log_file:
    for line in log_file:
        if "job_desc" in line:
            print(line)
            line = line.strip("job_desc:")
            # The first _ after each = should be changed to " "

            splitted_line = line.split("=")
            command = splitted_line[0]
            for token in splitted_line[1:]:
                value = token.split("_",1)[0]
                try:
                    key = token.split("_", 1)[1]
                except:
                    key=""
                command+="=%s %s"%(value,key)

            print(command.strip())
            exit(0)

print("Not found!")