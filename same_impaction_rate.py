import argparse
import csv
import os

#  Compute the intersection of max importance score between two lists
#  Input: action importance scores list (float list)
#  Output: list of indexes (int list)
def compare(a, b):
    a_max = max(a)
    b_max = max(b)

    a_idxs = [i for i in range(len(a)) if a[i] == a_max]
    b_idxs = [i for i in range(len(b)) if b[i] == b_max]

    intersection = [elm for elm in a_idxs if elm in b_idxs]
    return intersection

#  Change data type (str --> list) for a history
#  Input: history (str)
#  Output: history (float list)
def from_str_to_list(a):
    tmp_list = a[1:-1].split(', ')
    return [float(i) for i in tmp_list]

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', '--file', default="", help="History file", type=str, required=True)
    parser.add_argument('-strats', '--HXp_strategies', default="[exh, last_1, last_2, last_3, last_4, transition_1, transition_2, transition_3, transition_4]",
                        help="Exploration strategies for similarity measures", type=str, required=False)

    args = parser.parse_args()

    # Get arguments
    FILE = args.file
    STRATEGIES = args.HXp_strategies.split(', ')
    STRATEGIES[0] = STRATEGIES[0][1:]
    STRATEGIES[-1] = STRATEGIES[-1][:-1]

    # Init counters
    cpts = [0 for _ in range(len(STRATEGIES) - 1)]
    histories = 0

    abs_dir_path = os.getcwd()

    # Read FILE to compute the rate of same most important action returned by the HXP and approximate HXPs
    with open(abs_dir_path + os.sep + FILE, 'r') as f:
        reader = csv.reader(f)
        lines = []
        for row in reader:
            # Keep unchanged the line
            if (row[0] == 'History') or (row[0] == '' and row[2] != ''):
                lines.append(row)

            # Keep the line unchanged and compute order score
            else:
                histories += 1
                # Exh. importance scores
                exh = from_str_to_list(row[1])
                # Compare order with approximate scores
                for i in range(len(STRATEGIES) - 1):
                    approx = from_str_to_list(row[1 + 2 + (i * 2)])
                    # Update counter
                    if compare(exh, approx):
                        cpts[i] += 1
                lines.append(row)

        print('Number of histories: {}'.format(histories))
        print('Counters: {}'.format(cpts))

        # Add order score
        lines.append([''])  # Blank line
        line = ['Similar most important action (rate)', '', '']
        for i in range(len(STRATEGIES) - 1):
            line.append('')
            line.append(cpts[i] / histories)
        lines.append(line)

    # Write lines
    with open(abs_dir_path + os.sep + FILE, 'w') as f:
        writer = csv.writer(f)
        for row in lines:
            writer.writerow(row)
