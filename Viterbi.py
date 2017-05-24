#Jen Johnson
#CSCI321
#Problem 25 Viterbi

import numpy as np

def read(f):
    symbols_string = next(f).strip()

    division = next(f)
    string = next(f)

    symbol_set = []
    for char in string:
        char = char.strip()
        if char != "":
            symbol_set.append(char)

    division = next(f)
    string = next(f)
    state_set_string = string
    state_set = []

    for char in string:
        char = char.strip()
        if char != "":
            state_set.append(char)

    division = next(f)
    division = next(f)

    num_states = len(state_set)
    transmission = np.zeros((num_states, num_states), dtype = "float")

    row = 0
    while row < num_states:
        line = f.readline().strip()
        floats = line.split()
        for i in range (1, len(floats)):
            transmission[row][i-1] = floats[i]
        row+=1

    division = next(f)
    division = next(f)

    num_symbols = len(symbol_set)
    emission = np.zeros((num_states, num_symbols), dtype = "float")

    row = 0
    while row < num_states:
        line = f.readline().strip()
        floats = line.split()
        for i in range (1, len(floats)):
            emission[row][i-1] = floats[i]
        row+=1
        if not line:
            break

    return symbols_string, symbol_set, state_set, transmission, emission, state_set_string

def letter_to_index_in_set(letter, input_set):
    """ given a symbol or state, returns the index of that letter in its corresponding set"""
    i = 0
    while input_set[i] != letter:
        i+=1
    return i

def viterbi_initialize(symbols_string, symbol_set, state_set, emission):
    #set up the arrays
    vgraph = np.zeros((len(state_set), len(symbols_string)), dtype = "float")
    backtrack = np.zeros((len(state_set), len(symbols_string)), dtype = "int")

    #initialize the base cases
    initial_prob = 1/len(state_set)
    initial_symbol_index = letter_to_index_in_set(symbols_string[0], symbol_set)

    for i in range (0, len(state_set)):
        initial_emission = emission[i][initial_symbol_index]
        vgraph[i][0] = initial_prob * initial_emission

    return vgraph, backtrack

def get_result(vgraph, backtrack, state_set, symbols_string, incoming):
    """ uses backtrack matrix to get the result """

    end_probs = []

    for i in range (0, len(state_set)):
        end_probs.append(vgraph[i][len(symbols_string)-1])

    current_index = np.argmax(incoming)

    result = ""

    for i in range (len(symbols_string)-1, -1, -1):
        result+=state_set[current_index]
        current_index = backtrack[current_index][i]

    # reverse the string
    return result[::-1]

def Viterbi(symbols_string, symbol_set, state_set, transmission, emission):
    """ calls the Viterbi algorithm on the input """

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

            index = np.argmax(incoming)
            backtrack[j][i] = index
            vgraph[j][i] = incoming[index]

    result = get_result(vgraph, backtrack, state_set, symbols_string, incoming)

    return result

def main():
    f = open("ViterbiInput.txt")

    symbols_string, symbol_set, state_set, transmission, emission, state_set_string = read(f)

    result= Viterbi(symbols_string, symbol_set, state_set, transmission, emission)

    file = open("ViterbiOutput.txt", "w")
    file.write(result)
    file.close()

if __name__ == "__main__":
    main()
