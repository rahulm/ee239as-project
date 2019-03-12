import argparse
import subprocess
import csv

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('exp_csv', type=str,
        help="location of the csv containing experiment configurations.")
    args = parser.parse_args()
    return args

    
def main(args):
    print(args.exp_csv)
    with open(args.exp_csv, newline='') as exp_csv:
        reader = csv.DictReader(exp_csv)
        
        # get each experiment configuration
        for exp_i, exp_row in enumerate(reader):
            exp_cmd = ['python', 'eigenface_train.py']
            
            for k, v in exp_row.items():
                exp_cmd.append("--{}".format(k))
                exp_cmd.append(str(v))
            
            print("Starting experiment {}...\n{}\n".format(exp_i, exp_cmd))
            subprocess.run(exp_cmd)
            print("\nCompleted experiment {}...\n\n".format(exp_i))

if __name__ == '__main__':
    args = get_args()
    main(args)
    
    