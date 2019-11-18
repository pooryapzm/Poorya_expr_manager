import logging
from .remote_fetcher import RemoteFetcher


def setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    return logger

def fetch_expr_file_server_list(seed, server_list, file_list, force_fetch):
    logger = logging.getLogger()
    save_model_path = ""
    for potential_server in server_list:
        rf = RemoteFetcher(potential_server)
        transfered_files, save_model_path, error_in_connection = rf.fetch_expr_file(seed, file_list,
                                                                                    force_fetch=force_fetch)
        if error_in_connection:
            logger.info("Error in connection to %s" % potential_server)
        if transfered_files > 0:
            # out_file = "%s/weight_plot_%s.pdf" % (save_model_path, seed)
            break

    if save_model_path == "":
        logger.info("The experiment has not been found!")
    return save_model_path