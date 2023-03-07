#!/usr/bin/python
__author__ = "Jaiveer Singh"
__email__ = "jaiveer.singh@yale.edu"
__copyright__ = "Copyright 2023"
__license__ = "GPL"
__version__ = "1.0.0"

## Usage: python hw_1_final.py -i <input file> -s <score file>
## Example: python hw_1_final.py -i input.txt -s blosum62.txt
### Note: Smith-Waterman Algorithm

import argparse
import numpy as np
from enum import IntEnum
import pandas as pd

### This is one way to read in arguments in Python. 
parser = argparse.ArgumentParser(description='Smith-Waterman Algorithm')
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-s', '--score', help='score file', required=True)
parser.add_argument('-o', '--opengap', help='open gap', required=False, default=-2)
parser.add_argument('-e', '--extgap', help='extension gap', required=False, default=-1)
args = parser.parse_args()

#Create a class that is useful for connducting traceback
class Trace(IntEnum):
    STOP=0
    LEFT=1
    UP=2
    DIAGONAL=3

#This function reads in the blosum62.txt file into a 2D numpy array and returns it 
def loading_from_blosum_txt(scoreFile):
    blosum = np.zeros((23,23),dtype=int) #Create an empty 23 x 23 matrix to then fill in below
    file = open(scoreFile)
    blosum_l = file.readlines()

    #Looks at all of the numeric values in the blosum matrix and adds it to the created numpy array
    for blosum_l_n,blosum_ls in enumerate(blosum_l):
        if blosum_l_n != 0:
            blosum_vals = blosum_ls.split()
            for blosum_ind,blosum_num in enumerate(blosum_vals):
                if blosum_ind != 0:
                    blosum[blosum_ind-1][blosum_l_n-1] = int(blosum_num)
    return blosum

#This function matches the amino acids with their corresponding similarity score from the blosum matrix
def finding_similarity_score_value(aa1, aa2, blosum_matrix):
    #Creating a dictionary that represents which amino acids are paired with which indices
    str1 = "A, B, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, X, Y, Z"
    str2 = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
    keys = str1.split(", ") 
    values = str2.split(" ")
    dictionary = {}
    for i in range(len(keys)):
        dictionary[keys[i]] = values[i]

    #Using the created dictionary to match the letters to the indices and then return them
    i1 = int(dictionary[aa1])
    i2 = int(dictionary[aa2])
    return blosum_matrix[i1][i2]

#This function calculates the score matrix for the two given sequences based on the blosum62.txt matrix and the gap penalties
def running_smith_waterman(openGap, extGap, seq1, seq2, blosum, score_matrix):
    row_length = len(seq1)+1
    col_length = len(seq2)+1

    highest_score = -2
    index_highest_score = (-2,-2)

    optimal_matrix = np.zeros((row_length, col_length), dtype=int)

    up_penalty = []
    left_penalty = []

    #Calculating scores for the Smith-Waterman alignment (based on the function found on Wikipedia)
    for i in range(1, len(seq1)+1):
        for j in range(1, len(seq2)+1):
            #Look at the diagonal and add the value based on the blosum matrix
            blosum_score = finding_similarity_score_value(seq2[j-1], seq1[i-1], blosum)
            diag=score_matrix[i-1][j-1]+blosum_score
            #Look up and add the gaps (based on the open gap penalty and the gap extension penalties)
            for val in range(1, i+1):
                up_penalty.append(score_matrix[val][j]+openGap+(i-val-1)*extGap)
            up = max(up_penalty)
            #Look left and add the gaps (based on the open gap penalty and the gap extension penalties)
            for val in range(1, j+1):
                left_penalty.append(score_matrix[i][val]+openGap+(j-val-1)*extGap)
            left = max(left_penalty)
            #Determine the score based on the max of 0 and the diagonal, up, left values
            score_matrix[i][j]=max(
                0,
                diag,
                up,
                left
            )
            #Run the traceback_prep function to store which direction was the highest
            traceback_prep(score_matrix, optimal_matrix, i, j, diag, left, up)

            #Identify the index of the highest score in the matrix
            if score_matrix[i][j] >= highest_score:
                index_highest_score = (i, j)
                highest_score = score_matrix[i][j]
            
            up_penalty = []
            left_penalty = []
    
    sequnce_1_aligned, sequence_2_aligned = traceback(optimal_matrix, index_highest_score, seq1, seq2)

    return formatting_for_output(seq1, seq2, sequnce_1_aligned, sequence_2_aligned, index_highest_score)

#Based on the value that was assigned to the position, this function stores the move as diagonal, up, or left. Traceback stops if the value is 0. 
def traceback_prep(score_matrix, optimal_matrix, i, j, diag, left, up):
        if score_matrix[i][j]==0:
            optimal_matrix[i][j]=Trace.STOP

        elif score_matrix[i][j]==diag:
            optimal_matrix[i][j]=Trace.DIAGONAL

        elif score_matrix[i][j]==left:
            optimal_matrix[i][j]=Trace.LEFT

        elif score_matrix[i][j]==up:
            optimal_matrix[i][j]=Trace.UP

#This function takes the two aligned sequences after traceback and formats the alignment to resemble the output shown in the sample text files
def formatting_for_output(sequence1, sequence2, alignment1, alignment2, index_highest_score):
    len_s1 = len(sequence1)+1
    len_s2 = len(sequence2)+1
    row_index, col_index = index_highest_score

    #Add the closing parentheses to the aligned sequence and the rest of the sequence not aligned after it 
    alignment1+=')'+sequence1[row_index:len_s1]
    alignment2+=')'+sequence2[col_index:len_s2]

    alignment_c1=len(alignment1.replace("-",""))
    alignment_c2=len(alignment2.replace("-",""))
    
    #Add the opening parentheses to the aligned sequence and other parts of the sequence not aligned before it 
    alignment1=sequence1[0:(len_s1-alignment_c1)]+"("+alignment1
    alignment2=sequence2[0:(len_s2-alignment_c2)]+"("+alignment2

    alignment1, alignment2 = add_spaces_before_match(alignment1, alignment2)
    compared_string = add_matching_bars(alignment1, alignment2)

    return alignment1, compared_string, alignment2

def add_spaces_before_match(alignment1, alignment2):
    #Add spaces to alignment
    if len(alignment1)>=len(alignment2):
        added_spaces_f=alignment1.find('(')-alignment2.find('(')
        added_spaces_b=len(alignment1)-alignment1.find(')')-1

        alignment2=(" "*added_spaces_f)+alignment2+(" "*added_spaces_b)
    else:
        added_spaces_f=alignment2.find('(')-alignment1.find('(')
        added_spaces_b=len(alignment2)-alignment2.find(')')-1

        alignment1=(" "*added_spaces_f)+alignment1+(" "*added_spaces_b)
    
    return alignment1, alignment2

#This function adds bars to any matches between the two aligned sequences
def add_matching_bars(alignment1, alignment2):
    string_compared = ""

    if len(alignment1) == len(alignment2):
        for i in range(len(alignment1)):
            if (alignment1[i].isalpha() and alignment1[i] == alignment2[i]):
                string_compared+="|"

            else:
                string_compared+=" "
    
    return string_compared
    
#This function goes back through the optimal_matrix and chooses which direction to traceback
def traceback(optimal_matrix, index_highest_score, s1, s2):
    aligned_sequence1 = ""
    aligned_sequence2 = ""
    working_sequence1 = ""
    working_sequence2 = ""

    max_row, max_col = index_highest_score
    while optimal_matrix[max_row][max_col]!=0: #The condition for stopping
        if optimal_matrix[max_row][max_col]==Trace.DIAGONAL:
            working_sequence1=s1[max_row-1]
            working_sequence2=s2[max_col-1]

            max_row-=1
            max_col-=1
        elif optimal_matrix[max_row][max_col]==Trace.UP:
            working_sequence1=s1[max_row-1]
            working_sequence2='-'

            max_row-=1
        elif optimal_matrix[max_row][max_col]==Trace.LEFT:
            working_sequence1='-'
            working_sequence2=s2[max_col-1]

            max_col-=1
        
        aligned_sequence1+=working_sequence1
        aligned_sequence2+=working_sequence2

    #Flip the sequences
    aligned_sequence1 = aligned_sequence1[::-1]
    aligned_sequence2 = aligned_sequence2[::-1]
    return aligned_sequence1, aligned_sequence2

### Implement your Smith-Waterman Algorithm
def runSW(inputFile, scoreFile, openGap, extGap):
    with open("output.txt", "w") as f:
        print("-----------\n|Sequences|\n-----------", file = f)
        lines = open(inputFile).read().splitlines()

        s1 = lines[0]
        s2 = lines[1]

        print("sequence1\n" + s1, file = f) #Print sequence 1 (as shown in the sample output file)
        print("sequence2\n" + s2, file = f) #Print sequence 2 (as shown in the sample output file)

        #Create and initialize the score matrix with column and row of 0s
        score_matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)

        blosum = loading_from_blosum_txt(scoreFile)
        alignment1, string_compared, alignment2 = running_smith_waterman(openGap, extGap, s1, s2, blosum, score_matrix)

        #This function takes the calculated score matrix and formats it again in tab-delimited format (as shown in the sample output file)
        print("--------------\n|Score Matrix|\n--------------", file = f)
        seq1 = ' '+s1[:]
        seq2 = ' '+s2[:]
        df = pd.DataFrame(score_matrix.T, index = list(seq2), columns = list(seq1))
        print(df.to_csv(sep = '\t'), file = f, end = "")

        #Output the higest score and the alignment (as shown in the sample output file)
        highest_score = np.amax(score_matrix)
        print("----------------------\n|Best Local Alignment|\n----------------------", file=f)
        print("Alignment Score:" + str(highest_score), file=f) #The alignment score is the maximum score in the matrix
        print("Alignment Results:", file=f)
        print(alignment1, file=f)
        print(string_compared, file=f)
        print(alignment2, file=f)


### Run your Smith-Waterman Algorithm
runSW(args.input, args.score, args.opengap, args.extgap)

## Usage: python hw_1_final.py -i <input file> -s <score file>
## Example: python hw_1_final.py -i input.txt -s blosum62.txt
