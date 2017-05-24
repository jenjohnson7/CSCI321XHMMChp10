#Jen Johnson
#CSCI321
#XHMM

"""CNVs found:
SAMPLE CNV START END Q_EXACT
HG00121 DEL 22:18898402-18898541 22:18913201-18913235 7.759547224438718
HG00113 DUP 22:17071768-17071966 22:17073168-17073440 17.61165293623702
"""

from scipy.stats import norm
import numpy as np
from math import log10

def read(filename):

    # num_targets and targets
    f = open(filename)
    target_string = next(f)
    matrix_targets = target_string.split()
    targets = matrix_targets[1:]
    num_targets = len(targets)

    # num_samples and samples
    sample_strings = np.loadtxt(filename, dtype = str, skiprows=1, usecols =(0))
    num_samples = len(sample_strings)
    samples = []
    for sample in sample_strings:
        samples.append(sample[2:-1])

    # array to use all cols except the sample_num col
    temp = []
    for i in range (1, num_targets+1):
        temp.append(i)

    # read depths
    read_depths = np.loadtxt(filename, skiprows=1, usecols = temp)

    return num_targets, targets, num_samples, samples, read_depths

def get_transmission(p, q):
    """ 3 x 3 matrix
    States are Dup, Dip, Del
    Determined by Table 1 """

    transmission = np.zeros((3, 3), dtype = np.longdouble)
    transmission[0][0]=1-q
    transmission[0][1]=q
    transmission[1][0]=p
    transmission[1][1]=1-2*p
    transmission[1][2]=p
    transmission[2][1]=q
    transmission[2][2]=1-q

    return transmission

def get_emission(current_obs, current_col, current_state_index):
    """ given the observation array current_obs and the indices of observation of interest
    returns the emission probability of the current state using pdf function """

    local_obs = current_obs[current_col]

    mean = -1
    variance = 1

    if current_state_index == 0: #dup
        mean = 3
    elif current_state_index == 1: #dip
        mean = 0
    else: #del
        mean = -3

    #variance of 1
    return norm.pdf(local_obs, mean, variance)

def initialize_hmm(transmission, num_targets, p, current_obs):
    #set up the arrays
    vgraph = np.zeros((3, num_targets), dtype = np.longdouble)
    backtrack = np.zeros((3, num_targets), dtype = "int")

    #initialize the base cases.

    # dup and del
    #current_col = 0
    dup_del_initial_prob = p
    dup_initial_emission = get_emission(current_obs, 0, 0)
    del_initial_emission = get_emission(current_obs, 0, 2)
    vgraph[0][0] = dup_del_initial_prob * dup_initial_emission
    vgraph[2][0] = dup_del_initial_prob * del_initial_emission

    # dip
    dip_initial_prob = 1-2*p
    dip_initial_emisison = get_emission(current_obs, 0, 1)
    dip_initial = dip_initial_prob * dip_initial_emisison
    vgraph[1][0] = dip_initial

    return vgraph, backtrack

def viterbi_changing_emissions(vgraph, backtrack, current_obs, transmission, num_targets):
    """ calls version of viterbi where emissions are recalculated at each iteration """

    for i in range (1, num_targets-1):
        for j in range (0, 3):

            incoming = []
            current_emission = get_emission(current_obs, i, j)

            for k in range (0, 3):
                current_transmission = transmission[k][j]
                incoming.append(current_transmission * current_emission * vgraph[k][i-1])

            index = np.argmax(incoming)
            backtrack[j][i] = index
            vgraph[j][i] = incoming[index]

    state_type, important_indices = get_cnv(backtrack, incoming, vgraph, num_targets)

    return state_type, important_indices

def get_cnv(backtrack, incoming, vgraph, num_targets):
    """ uses backtrack and incoming to find if/where a cnv occurs in the sample
    returns the state_type (0/DUP or 2/DEL)
    and the indices where the cnv occurs """

    end_probs = []

    for i in range (0, 3):
        end_probs.append(vgraph[i][len(backtrack)-1])

    # viterbi: get the MAX prob going into sink
    current_index = np.argmax(incoming)

    # initialize results to compare to and structures in which to store results
    current_state = str(current_index)
    important_indices = []
    state_array = []

    # starting from end of backtrack, compare states
    for i in range (num_targets-2, -1, -1):
        new_state = str(current_index)
        if new_state != current_state:
            # if there has been a change
            important_indices.append(i)
            state_array.append(new_state)
        # update the for loop
        current_index = backtrack[current_index][i]
        current_state = new_state

    state_type = -1

    # if a cnv was detected, there will be 2 states: one dip, and one other
    if len(state_array) != 0:
        state_array.remove('1') # remove dip
        state_type = int(state_array[0]) # get the other

    return state_type, important_indices

def get_cnv_type(state_type):
    """ converts int state to string state """
    result = ""
    if state_type == 0:
        result = "DUP"
    elif state_type == 2:
        result = "DEL"
    return result

def get_strings(i, samples, state_type, important_indices, targets):
    """ get strings for printing given input """

    sample_string = samples[i]
    cnv_string = get_cnv_type(state_type)
    start_index = important_indices[1]
    start_string = targets[start_index+1]
    end_index = important_indices[0]
    end_string = targets[end_index]

    return sample_string, cnv_string, start_string, end_string

def string_prob_changing_emissions(vgraph, current_obs, transmission, num_targets):
    """ calls version of string prob where emissions are recalculated at each iteration """

    # fill the arrays
    for i in range (1, num_targets):
        for j in range (0, 3):

            incoming = []
            current_emission = get_emission(current_obs, i, j)

            for k in range (0, 3):
                current_transmission = transmission[k][j]
                incoming.append(current_transmission * current_emission * vgraph[k][i-1])

            vgraph[j][i] = sum(incoming)

    result = 0
    # forward(sink) == sum of the last col
    #sum over all i of vgraph[i][last col]
    for x in range (0, 3):
        temp = vgraph[x][num_targets-1]
        result +=temp

    return result, vgraph

def string_prob_changing_emissions_backward(current_obs, transmission, num_targets):
    """ calls backward version of string prob where emissions are recalculated at each iteration """

    vgraph = np.zeros((3, num_targets), dtype = np.longdouble)

    #initialize base cases
    for i in range (0, 3):
        vgraph[i][len(current_obs)-1] = 1

    #fill the arrays
    for i in range (len(current_obs)-2, -1, -1):
        for j in range (0, 3):
            outgoing = []

            for k in range (0, 3):
                current_transmission = transmission[j][k]
                current_emission = get_emission(current_obs, i+1, k)
                outgoing.append(current_transmission * current_emission * vgraph[k][i+1])

            vgraph[j][i]=sum(outgoing)

    return vgraph

def get_path(transmission, current_obs, important_indices, int_state_type):
    """ returns the probability of the path with state int_state_type using transmission and emission """

    start = important_indices[1]+1
    end = important_indices[0]
    distance = end - start

    path_transmission = 1
    # calculating prob of the cnv of type int_state_type
    # transmission will not change
    # multiply transmission value by itself (distance) number of times
    transmission_type = transmission[int_state_type][int_state_type]
    for i in range (start+1, end+1):
        path_transmission *= transmission_type

    path_emission = 1
    # emission will change
    # get the emission for that column for the state (int_state_type)
    for i in range (start+1, end+1):
        path_emission *= get_emission(current_obs, i, int_state_type)

    result = path_emission * path_transmission

    return result

def get_q_exact(vgraph, current_obs, transmission, num_targets, important_indices, int_state_type):
    """ calculates q exact using string_prob_changing_emissions (forward and backward) and get_path """

    denominator, vgraph = string_prob_changing_emissions(vgraph, current_obs, transmission, num_targets)

    forward_start = vgraph[int_state_type][important_indices[1]+1]

    backward_matrix = string_prob_changing_emissions_backward(current_obs, transmission, num_targets)

    backward_end = backward_matrix[int_state_type][important_indices[0]]

    pr_path = get_path(transmission, current_obs, important_indices, int_state_type)

    numerator = forward_start * backward_end * pr_path

    q_exact = numerator/denominator

    return q_exact

def phred(q_exact):
    x = log10(1-q_exact)
    result = -10 * x
    return result

def main():

    num_targets, targets, num_samples, samples, read_depths = read("XHMM.in.txt")

    p = 1e-8
    q = 1/6

    transmission = get_transmission(p, q)

    file = open("XHMM.out.txt", "w")
    title_string = ("SAMPLE CNV START END Q_EXACT\n")
    file.write(title_string)

    for i in range (0, len(read_depths)-1):
        # call viterbi/find cnvs on EACH sample

        current_obs = read_depths[i]

        vgraph, backtrack = initialize_hmm(transmission, num_targets, p, current_obs)

        state_type, important_indices = viterbi_changing_emissions(vgraph, backtrack, current_obs, transmission, num_targets)

        # if a cnv was found
        if len(important_indices) != 0:

            sample_string, cnv_string, start_string, end_string = get_strings(i, samples, state_type, important_indices, targets)

            vgraph, backtrack = initialize_hmm(transmission, num_targets, p, current_obs)

            q_exact = get_q_exact(vgraph, current_obs, transmission, num_targets, important_indices, state_type)

            result = phred(q_exact)

            body_string = sample_string + " " + cnv_string + " " + start_string + " " + end_string + " " + str(result)
            file.write(body_string + "\n")

    file.close()

if __name__ == "__main__":
    main()
