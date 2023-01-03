# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def replace_first_line( src_filename, target_filename):
    f = open(src_filename)
    first_line, remainder = f.readline(), f.read()
    t = open(target_filename,"w")
    t.write(first_line.lstrip(',') + "\n")
    t.write(remainder)
    t.close()

import pandas as pd

df = pd.read_csv('PRJNA297261.tsv', sep='\t')

sequences = df['AASeq']
df2 = pd.DataFrame()
df2['AASeq'] = sequences

aux = df[df['RunId'] == 'SRR2549140']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549140'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549130']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549130'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549131']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549131'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549132']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549132'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549133']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549133'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549127']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549127'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549128']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549128'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549129']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549129'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549147']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549147'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549146']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549146'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549144']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549144'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549145']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549145'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549139']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549139'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549136']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549136'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549143']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549143'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549142']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549142'})
df2 = df2.fillna(0)

aux = df[df['RunId'] == 'SRR2549141']
aux = aux[['AASeq', 'cloneFraction']]
df2 = pd.merge(df2, aux, on='AASeq', how= 'left')
df2 = df2.rename(columns={"cloneFraction": 'SRR2549141'})
df2 = df2.fillna(0)

cellType  = ['TReg',       'TReg',       'TConv',      'TReg',       'TReg',       'TConv',      'TReg',       'TConv',      'TReg',       'TConv',      'TReg',       'TReg',       'TConv',      'TReg',       'TReg',       'TConv',      'TReg']
condition = ['BC',         'BC',         'BC',         'BC',         'BC',         'Healthy',    'Healthy',    'BC',         'BC',         'BC',         'BC',         'BC',         'Healthy',    'BC',         'Healthy',    'Healthy',    'Healthy']
df2 = df2.loc[~(df==0).all(axis=1)]
df2 = df2.T
df2.columns = df2.iloc[0]        
df2.drop(index = df2.index[0], axis=0, inplace=True) 
df2['label'] = condition
df2['cellType'] = cellType

df_TReg = df2[df2['cellType'] == 'TReg']
df_TReg = df_TReg.loc[:, (df_TReg != 0).any(axis=0)]
df_TReg = df_TReg.drop('cellType', axis=1)
df_TReg.to_csv('TRegs.csv', sep=',')
replace_first_line( 'TRegs.csv', 'TRegs.csv')

df_TConv = df2[df2['cellType'] == 'TConv'] 
df_TConv = df_TConv.loc[:, (df_TConv != 0).any(axis=0)]
df_TConv = df_TConv.drop('cellType', axis=1)
df_TConv.to_csv('TConvs.csv', sep=',')
replace_first_line( 'TConvs.csv', 'TConvs.csv')

