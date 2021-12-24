import os
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger(object):

    def __init__(self, path):
        self.path = path
        self.writer = SummaryWriter(log_dir=path)        
        self.tb_values = {}
        self.csv_values = {}
        self.csv_logger = None

    def record(self, key, value):
        self.tb_values[key] = value
        self.csv_values[key] = value

    def dump(self, step, dump_csv=False):
        for k in self.tb_values.keys():
            self.writer.add_scalar(k, self.tb_values[k], step)
        self.writer.flush()
        # TB values get cleared every time.
        self.tb_values.clear()
        if dump_csv:
            if self.csv_logger is None:
                self.csv_file_handler = open(os.path.join(self.path, "log.csv"), "wt")
                self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(self.csv_values.keys()))
                self.csv_logger.writeheader()
            self.csv_logger.writerow(self.csv_values)
            self.csv_file_handler.flush()
