from sklearn.metrics import precision_score, recall_score, f1_score       
from scipy.special import softmax
from sklearn.metrics import accuracy_score
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
import datetime


def setup_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    directory = "D:/Xelp_work/FSL Project/API_Demos/Lookup_Member/"
    error_file = os.path.join(directory, "member_errorlog.txt")
    handler = TimedRotatingFileHandler(error_file, when="midnight", interval=1, backupCount=15)
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

def log_summary(logger):
    try:
        log_directory = "D:/Xelp_work/FSL Project/API_Demos/Lookup_Member/"
        summary_file = os.path.join(log_directory, "log_summary.txt")
        today = datetime.date.today().strftime("%Y-%m-%d")
        
        # Initialize counters for requests, success, and errors
        total_requests = 0
        success_count = 0
        error_count = 0
        
        # Iterate through the log messages
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                with open(handler.baseFilename, 'r') as log_file:
                    lines = log_file.readlines()
                    for line in lines:
                        total_requests += 1
                        if "success" in line.lower():
                            success_count += 1
                        elif "error" in line.lower():
                            error_count += 1
        
        # Write the summary to the log summary file
        with open(summary_file, "a") as f:
            f.write(f"Summary for {today}\n")
            f.write(f"Total Requests: {total_requests}\n")
            f.write(f"Success Count: {success_count}\n")
            f.write(f"Error Count: {error_count}\n")
            f.write("\n")
    except Exception as exp:
        logger.exception(exp)

def create_seq1(train):
    train = train.astype('str')
    sep ="[SEP]"
    return sep+ train['Ami_'] + sep + train['EmpDepLname_'] + sep + train['EmpDepFname_'] + \
            sep + train['MemberDob_'] + sep + train['EmpLastNm_'] + sep + train['EmpFirstNm_'] + \
            sep + train['EmpDob_'] + sep + train['EmpInd_'] + sep + train['SubLastName_'] + sep + train['SubFirstName_'] + \
            sep + train['SubDob_'] + sep + train['AmiNum_'] + sep + train['DepLastName_'] + sep + train['DepFirstName_'] +\
            sep + train['DepDob_'] + sep + train['SubSsn_']

def create_seq2(val):
    val = val.astype('str')
    sep ="[SEP]"
    return sep + val['Ami'] + sep + val['EmpDepLname'] + sep + val['EmpDepFname'] + \
        sep + val['MemberDob'] + sep + val['EmpLastNm'] + sep + val['EmpFirstNm'] + \
        sep + val['EmpDob'] + sep + val['EmpInd'] + sep + val['SubLastName'] + sep + val['SubFirstName'] + \
        sep + val['SubDob'] + sep + val['AmiNum'] + sep + val['DepLastName'] + sep + val['DepFirstName'] +\
        sep + val['DepDob'] + sep + val['SubSsn']
