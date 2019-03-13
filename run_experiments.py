import argparse
import subprocess
import csv

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_csv', type=str,
        help="location of the csv containing experiment configurations.")
    parser.add_argument('--exp_log', type=str,
        help="location of log file detailing a fail or complete status for experiments")
    args = parser.parse_args()
    return args

    
def main(args):
    print(args.exp_csv)
    print(args.exp_log)
    with open(args.exp_log, mode='a', newline='') as exp_log:
        with open(args.exp_csv, mode='r', newline='') as exp_csv:
            reader = csv.DictReader(exp_csv)
            # get each experiment configuration
            for exp_i, exp_row in enumerate(reader):
                exp_cmd = ['python', 'eigenface_train.py']
                
                for k, v in exp_row.items():
                    exp_cmd.append("--{}".format(k))
                    exp_cmd.append(str(v))
                
                print("Starting experiment {}...\n{}\n".format(exp_i, exp_cmd))
                process = subprocess.run(exp_cmd)
                exit_code = process.returncode
                if exit_code == 0:
                    print("\nCompleted experiment {}...\n\n".format(exp_i))
                    exp_log.write("SUCCESS - exit code {} - experiment {}...\n{}\n".format(exit_code, exp_i, exp_cmd))
                else:
                    print("\nFailed experiment {}...\n\n".format(exp_i))
                    exp_log.write("FAIL - exit code {} - experiment {}...\n{}\n".format(exit_code, exp_i, exp_cmd))
                
                exp_log.flush()


if __name__ == '__main__':
    args = get_args()
    main(args)
    
    