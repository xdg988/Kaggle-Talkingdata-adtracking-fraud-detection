# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print("Reading the data...\n")
df1 = pd.read_csv('../input/097960/sub_it5.csv')
df2 = pd.read_csv('../input/097960/sub_it7.csv')

models = { 'df1' : {
                    'name':'1',
                    'score':97.69,
                    'df':df1 },
           'df2' :{'name':'2',
                    'score':97.33,
                    'df':df2 },
         }

df1.head()         

isa_lg = 0
isa_hm = 0
isa_am = 0
isa_gm=0
print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
    isa_am +=isa_am
    isa_gm *= isa_gm
isa_lg = np.exp(isa_lg/2)
isa_hm = 2/isa_hm
isa_am = isa_am/2
isa_gm = (isa_gm)**(1/2)

print("Isa log\n")
print(isa_lg[:2])
print()
print("Isa harmo\n")
print(isa_hm[:])

sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()

sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()

sub_fin=pd.DataFrame()
sub_fin['click_id']=df1['click_id']
sub_fin['is_attributed']= (5*isa_lg+3*isa_hm+2*isa_am)/10

print("Writing...")
sub_log.to_csv('submission_log2.csv', index=False, float_format='%.9f')
#sub_hm.to_csv('submission_hm2.csv', index=False, float_format='%.9f')
#sub_fin.to_csv('submission_final.csv', index=False, float_format='%.9f')
