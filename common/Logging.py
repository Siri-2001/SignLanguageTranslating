import logging,datetime
import os.path


class Log:
    def __init__(self, msg, logging_path_file):
        self.msg = msg
        self.lpf = logging_path_file

    def log(self):

        logging.basicConfig(filemode ="w", filename=self.lpf, level=0)
        logging.info(self.msg)
        print(self.msg)