# -*- coding: utf-8 -*-
"""
This script will get the necessary columns and concatenate the files in AIRR format to create a new file in which can be used by the data_preprocess.py script. 
Thus creating the input file for GENTLE
"""

def translate(seq):
	table = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
	           'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
	           'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
	           'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
	           'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
	           'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
	           'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
	           'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
	           'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
	           'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
	           'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
	           'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
	           'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
	           'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
	           'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
	           'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
           }
	protein = ""
	for i in range(0, len(seq), 3):
		codon = seq[i:i + 3]
		if len(codon) < 3:
			break
		allowed_s = 'ACTG'
		if all(ch in allowed_s for ch in codon):
			protein+= table[codon]
		
	
	return protein 
	
import pandas as pd
import math

df = pd.read_csv('HD13M_fmt7_db-pass.tsv', sep='\t')  # upload the AIRR file with sequence, and counts

aux = df[['sequence_id', 'sequence', 'consensus_count']] # get only the necessary columns
aux['label'] = 'X' # create and additional column with the label of the sample

df = pd.read_csv('HD13M_fmt19.tsv', sep='\t')  # upload the AIRR file with CDR3 info

aux2 = df[['rearrangement_id','cdr3_start', 'cdr3_end']] # get only the necessary columns

new_id = []
for (i, row) in aux2.iterrows():

	if row[0].split('|')[0] == 'reversed':
		new_id.append(row[0].split('|')[1])
	else:
		new_id.append(row[0].split('|')[0])
		
aux2['sequence_id'] = new_id	
aux2 = aux2.drop('rearrangement_id', axis=1)

df = pd.merge(aux, aux2, on='sequence_id', how= 'left')

aa = []
for (i, row) in df.iterrows():
    if not math.isnan(row[4]) and not math.isnan(row[5]):
    	aa.append(translate(row[1][int(row[4]):int(row[5])]))
    else:
    	aa.append('not')

	
df['aa_seq'] = aa

df = df.drop(df[df.aa_seq == 'not'].index)


"""
Repeat the code as many times as the number of AIRR files you need to analyse
Using concatenation for each new AIRR file

new_df = pd.read_csv('new_file_with_counts.tsv', sep='\t')  # upload the AIRR file with sequence, and counts

aux = df[['sequence_id', 'sequence', 'consensus_count']] # get only the necessary columns
aux['label'] = 'X' # create and additional column with the label of the sample

df = pd.read_csv('new_file_with_cdr3_info.tsv', sep='\t')  # upload the AIRR file with CDR3 info

aux2 = df[['rearrangement_id','cdr3_start', 'cdr3_end']] # get only the necessary columns

new_id = []
for (i, row) in aux2.iterrows():

	if row[0].split('|')[0] == 'reversed':
		new_id.append(row[0].split('|')[1])
	else:
		new_id.append(row[0].split('|')[0])
		
aux2['sequence_id'] = new_id	
aux2 = aux2.drop('rearrangement_id', axis=1)

new_df = pd.merge(aux, aux2, on='sequence_id', how= 'left')

aa = []
for (i, row) in df.iterrows():
    if not math.isnan(row[4]) and not math.isnan(row[5]):
    	aa.append(translate(row[1][int(row[4]):int(row[5])]))
    else:
    	aa.append('not')

	
df['aa_seq'] = aa

df = df.drop(df[df.aa_seq == 'not'].index)

df = pd.concat([aux, aux2]) # concatenate the files
"""

df.to_csv('concated_AIRR.csv', sep=',') # save final file




