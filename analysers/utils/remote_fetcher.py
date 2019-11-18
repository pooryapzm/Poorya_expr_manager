import os
import paramiko
import yaml
import ntpath
import fnmatch
import warnings
import logging

config_file = os.path.dirname(os.path.abspath(__file__))+"/fetcher_config.yaml"


class RemoteFetcher:

    def __init__(self, server_name="m3"):
        with open(config_file, 'r') as stream:
            try:
                self.configs = yaml.load(stream)
            except yaml.YAMLError as exc:
                raise AssertionError("Cannot read config file. It should be in YAML format.")
        self.client = None
        self.server_name= server_name+"_server_info"
        warnings.simplefilter(action="ignore")

        # self.sftp = self.client.open_sftp()

    def _maybe_close_connection(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def _maybe_establish_connection(self):
        logger = logging.getLogger()
        if self.client is None:
            try:
                logger.info("Connecting to %s"%self.server_name)
                server_info = self.configs[self.server_name]
                proxy = None
                self.client = paramiko.SSHClient()
                self.client.load_system_host_keys()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.client.connect(server_info["server"], username=server_info["user"], password=server_info["pass"],
                                    sock=proxy)
                return True
            except:
                self.client=None
                return False
        else:
            return True

    # def _is_folder(self, item):
    #     lstatout = str(self.sftp.lstat(item)).split()[0]
    #     if 'd' in lstatout:
    #         return True
    #     else:
    #         return False

    def _find_expr_folder(self, seed):
        expr_dir_path = ""
        for remote_dir in self.configs[self.server_name]["remote_dirs"]:
            stdin, stdout, stderr = self.client.exec_command("cd %s; find . -name \"*%s*\"" % (remote_dir, seed))
            result_list = stdout.readlines()
            if len(result_list) > 0:
                relative_path = result_list[0].strip()
                expr_dir_path = remote_dir + "/" + relative_path
                break
        return expr_dir_path

    # This function is borrowed from ONMT code
    def _check_save_files_path(self, seed):
        dir = self.configs["local_folder"] + "/" + str(seed)
        save_model_path = os.path.abspath(dir)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        return save_model_path

    def _check_local_existence(self, seed, files_list):
        transfered_files = 0
        dir = self.configs["local_folder"] + "/" + str(seed)
        save_model_path = os.path.abspath(dir)
        if os.path.exists(save_model_path):
            saved_files = os.listdir(save_model_path)
            find_all = True
            for pattern in files_list:
                def _match_pattern(string, pattern):
                    return fnmatch.fnmatch(string, pattern)

                matches = [x for x in saved_files if _match_pattern(x, pattern)]
                if len(matches) == 0:
                    find_all = False
                transfered_files += len(matches)
        else:
            find_all = False

        return find_all, transfered_files, save_model_path

    def fetch_expr_file(self, seed, files_list, force_fetch=False, keep_connection=False):
        logger = logging.getLogger()
        save_model_path = ""
        is_find_all = False
        error_in_connection=False
        if not force_fetch:
            is_find_all, transfered_files, save_model_path = self._check_local_existence(seed, files_list)
        if not is_find_all:
            transfered_files = 0
            is_established = self._maybe_establish_connection()
            if is_established:
                expr_dir_path = self._find_expr_folder(seed)
                if expr_dir_path != "":
                    logger.info("Fetching file(s) from remote server (%s)"%self.server_name)
                    save_model_path = self._check_save_files_path(seed)
                    ftp_client = self.client.open_sftp()
                    for file in files_list:
                        stdin, stdout, stderr = self.client.exec_command(
                            "cd %s; find . -name \"*%s*\"" % (expr_dir_path, file))
                        result_list = stdout.readlines()
                        for result in result_list:
                            remote_file_abs_path = expr_dir_path + "/" + result.strip()
                            local_file_abs_path = save_model_path + "/" + ntpath.basename(result.strip())
                            ftp_client.get(remote_file_abs_path, local_file_abs_path)
                            transfered_files += 1
                    ftp_client.close()
            else:
                error_in_connection=True
        else:
            logger.info("%s files has been found in local folders"%str(seed))
        if not keep_connection:
            self._maybe_close_connection()

        return transfered_files, save_model_path, error_in_connection

