#!/usr/bin/env python
# coding: utf-8

# In[52]:


import cv2
import scipy
import nltk
import json
import os
import string
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
import numpy as np
from scipy import misc # feel free to use another image loader


# In[53]:


descriptions_file = "clean.json"


# In[54]:


with open(descriptions_file, "r") as j_desc:
    annos = json.load(j_desc)


# In[76]:


image_list=[]
text_list=[]
vocabulary=['\pad']

for annot in annos:
    rel_img_path = annot["image"][:-3]
    img_file = os.path.join("lfw", rel_img_path+"jpg")
    img = misc.imread(img_file)
    if img is not None:
        descriptions = annot["descriptions"]
        # print all the descriptions:
        #print("\nImage_Id:", annot["img_id"])
        description_list=[]
        
        for (i, description) in enumerate(descriptions, 1):
            #print("%d.) %s" %(i, description["text"]
            x=description["text"].lower()
            x=x.translate(string.punctuation)
            x=x.replace(',','')
            x=x.replace('(','')
            x=x.replace(')','')
            x=x.replace('/',' ')
            x=x.replace('+',' ')

            x=x.replace('.','')
            entry=x.split(' ')
            cleaned_tex=''
            for index,word in enumerate(entry):
                stemmed = porter.stem(word)
                if stemmed not in vocabulary and stemmed!='':vocabulary.append(stemmed)
                if(index==0):cleaned_tex=cleaned_tex+stemmed  
                else:cleaned_tex=cleaned_tex+' '+stemmed  
            #creating a vocabulary
            description_list.append(cleaned_tex)
        image_list.append(misc.imread(img_file))
        text_list.append(description_list)


# In[77]:


image_list=np.array(image_list)
image_list.shape


# In[78]:


import matplotlib.pyplot as plt


# In[124]:


numeric_vector=[]
for description in text_list:
    
    sub_numeric=[]
    for each_description in description:
        x=each_description.split(' ')
        vector = [0 for x in range(0,92)]
        for location,word in enumerate(x):
            try:
                vector[location]=vocabulary.index(word)
            except:
                pass
        vector = np.array(vector).astype('int16')
        sub_numeric.append((vector))
    
    numeric_vector.append(np.array(sub_numeric).astype('int16'))


# In[125]:


numeric_vector=np.array(numeric_vector)


# In[ ]:





# In[128]:





# In[129]:


plt.imshow(image_list[1])
plt.imshow(image_list[1])

for x in numeric_vector[1][0]:
    print(x,vocabulary[x])


# In[ ]:




