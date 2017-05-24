#Jen Johnson
#CSCI321
#Problem 27 Posterior

import numpy as np
from Viterbi import read, letter_to_index_in_set
from ProbString import prob_string

def backward_viterbi(symbols_string, symbol_set, state_set, transmission, emission):
    """ calculates the backward vgraph matrix"""

    vgraph = np.zeros((len(state_set), len(symbols_string)), dtype = "float")

    # initialize base cases
    for i in range (0, len(state_set)):
        vgraph[i][len(symbols_string)-1] = 1

    # fill the arrays
    for i in range (len(symbols_string)-2, -1, -1):
        for j in range (0, len(state_set)):
            outgoing = []
            current_state_index = letter_to_index_in_set(state_set[j], state_set)
            following_symbol_index = letter_to_index_in_set(symbols_string[i+1], symbol_set)

            for k in range (0, len(state_set)):
                following_state_index = letter_to_index_in_set(state_set[k], state_set)
                current_transmission = transmission[current_state_index][following_state_index]
                current_emission = emission[following_state_index][following_symbol_index]

                outgoing.append(current_transmission * current_emission * vgraph[k][i+1])

            vgraph[j][i]=sum(outgoing)

    return vgraph

def posterior(symbols_string, symbol_set, state_set, transmission, emission):
    """ calls forward-backward algorithm on the input """

    # call viterbi to get forward(sink) and vgraph matrix
    total_forward_prob, vgraph = prob_string(symbols_string, symbol_set, state_set, transmission, emission)

    # call viterbi to get backward vgraph matrix
    backward = backward_viterbi(symbols_string, symbol_set, state_set, transmission, emission)

    # elt wise multiplication
    matrix = np.multiply(vgraph, backward)

    # divide by total probability
    result = np.divide(matrix, total_forward_prob)

    return result

def main():
    f = open("PosteriorInput.txt")

    symbols_string, symbol_set, state_set, transmission, emission, state_set_string = read(f)

    result = posterior(symbols_string, symbol_set, state_set, transmission, emission)

    # found to 4 decimals
    result2 = np.around(result, 4)

    #transpose for easier printing
    final_result = result2.transpose()

    file = open("PosteriorOutput.txt", "w")
    file.write(state_set_string)
    for line in final_result:
        for col in line:
            file.write(str(col) + "\t")
        file.write("\n")

    file.close()

if __name__ == "__main__":
    main()
