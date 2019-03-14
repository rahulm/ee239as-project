import argparse
import subprocess
import csv
import sys

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    required_group = parser.add_argument_group('required arguments:')
    required_group.add_argument('--exp_csv', type=str, required=True,
        help="location of the csv containing experiment configurations.")
    required_group.add_argument('--log', type=str, required=True,
        help="location of where to output a log file detailing a fail or complete status for experiments")
    
    parser.add_argument('--pycmd', type=str, default='python',
        help="Python prompt to use when starting each experiment. Ex: 'python' vs 'python3'.")
    
    args = parser.parse_args()
    return args

def setup_custom_logging(log_filename):
    outfile = open(log_filename, 'a')
    
    class CustomLogging:
        def __init__(self, orig_stream):
            self.orig_stream = orig_stream
            self.fileout = outfile
        def write(self, data):
            self.orig_stream.write(data)
            self.orig_stream.flush()
            self.fileout.write(data)
            self.fileout.flush()
        def flush(self):
            self.orig_stream.flush()
            self.fileout.flush()
    
    sys.stdout = CustomLogging(sys.stdout)
    
def main(args):
    print(args.exp_csv)
    print(args.log)
    
    setup_custom_logging(args.log)
    print("Running run_experiments.csv with arguments:\n{}\n\n".format(args))

    with open(args.exp_csv, mode='r', newline='') as exp_csv:
        reader = csv.DictReader(exp_csv)
        # get each experiment configuration
        for exp_i, exp_row in enumerate(reader):
            exp_cmd = [args.pycmd, 'eigenface_train.py']
            
            for k, v in exp_row.items():
                exp_cmd.append("--{}".format(k))
                exp_cmd.append(str(v))
            
            print("Starting experiment {}...\n{}\n".format(exp_i, exp_cmd))
            process = subprocess.run(exp_cmd)
            exit_code = process.returncode
            if exit_code == 0:
                print("\nSUCCESS - exit code {} - experiment {}:\n{}\n".format(exit_code, exp_i, exp_cmd))
                print("\nCompleted experiment {}...\n\n".format(exp_i))
            else:
                print("\nFAIL - exit code {} - experiment {}:\n{}\n".format(exit_code, exp_i, exp_cmd))
                print("\nFailed experiment {}...\n\n".format(exp_i))


if __name__ == '__main__':
    args = get_args()
    main(args)
    
    