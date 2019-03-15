import argparse
import subprocess
import csv
import sys

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ranges_csv', type=str, required=True,
        help="location of the csv containing experiment configuration ranges.")
    parser.add_argument('--exp_config_out', type=str, required=True,
        help="location of where to output a csv of the configs")
    args = parser.parse_args()
    return args

def getAllSets(paramRanges, paramsRemaining):
    if len(paramsRemaining) == 0:
        return [[]]
    
    nextSets = getAllSets(paramRanges, paramsRemaining[1:])
    
    currParam = paramsRemaining[0]
    currRange = paramRanges[currParam]
    
    combinations = []
    for v in currRange:
        for nextSet in nextSets:
            combinations.append([v] + list(nextSet))
    
    return combinations


def main(args):
    
    hParamRanges = {}
    
    with open(args.ranges_csv, mode='r', newline='') as rangesCsv:
        reader = csv.DictReader(rangesCsv)
        
        # get list of hyperparameter ranges
        for row in reader:
            for k, v in row.items():
                if v != "":
                    if not (k in hParamRanges):
                        hParamRanges[k] = []
                    hParamRanges[k].append(v)
            
        
    # make list of rows to write
    hParamNames = list(hParamRanges.keys())
    
    allParamConfigs = getAllSets(hParamRanges, hParamNames)
    
    
    config_dir = os.path.dirname(args.exp_config_out)
    if (config_dir != "") and (not os.path.exists(config_dir)):
        os.makedirs(config_dir)
    
    hParamNames = ['exp_name'] + hParamNames
    with open(args.exp_config_out, 'w', newline='') as configCsv:
        writer = csv.writer(configCsv)
        writer.writerow(hParamNames)
        
        for expI, hParams in enumerate(allParamConfigs):
            writer.writerow(["exp_{}".format(expI)] + hParams)
    
    
if __name__ == '__main__':
    args = get_args()
    main(args)
