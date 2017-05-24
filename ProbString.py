#Jen Johnson
#CSCI321
#Problem 26 ProbString

import numpy as np
from Viterbi import read, letter_to_index_in_set, viterbi_initialize

def prob_string(symbols_string, symbol_set, state_set, transmission, emission):
    """" calls prob_string on input """

    # initialization
    vgraph, backtrack = viterbi_initialize(symbols_string, symbol_set, state_set, emission)

    #fill the arrays
    for i in range (1, len(symbols_string)):
        for j in range (0, len(state_set)):
            incoming = []
            current_state_index = letter_to_index_in_set(state_set[j], state_set)
            current_symbol_index = letter_to_index_in_set(symbols_string[i], symbol_set)
            current_emission = emission[current_state_index][current_symbol_index]

            for k in range (0, len(state_set)):
                prev_state_index = letter_to_index_in_set(state_set[k], state_set)
                current_transmission = transmission[prev_state_index][current_state_index]

                incoming.append(current_transmission * current_emission * vgraph[k][i-1])
            
            # use sum instead of max
            vgraph[j][i] = sum(incoming)

    result = 0

    #sum over all i of vgraph[i][len(symbols_string)]
    for k in range (0, len(state_set)):
        result += vgraph[k][len(symbols_string)-1]

    return result, vgraph

def main():
    f = open("ProbStringInput.txt")

    symbols_string, symbol_set, state_set, transmission, emission, state_set_string = read(f)

    result, vgraph = prob_string(symbols_string, symbol_set, state_set, transmission, emission)

    file = open("ProbStringOutput.txt", "w")
    file.write(str(result))
    file.close()

if __name__ == "__main__":
    main()
