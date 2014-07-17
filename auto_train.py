#!/usr/bin/python3

import os
import re
import digit_recognizer

def get_last_backup():
    listing = [k for k in os.listdir('.') if re.match('^backup_(\d+(\.\d+)?)\.pkl$', k)]
    backup, *_ = sorted(listing, key=lambda k: float(re.match('^backup_(\d+(\.\d+)?)\.pkl$', k).group(1)))
    return backup

if __name__ == '__main__':
    config_file = get_last_backup()
    print("Config File: {}".format(config_file))
    digit_recognizer.teacher(config_file=config_file, clear=False, save=False)
