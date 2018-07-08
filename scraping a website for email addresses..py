
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
import requests
page_link ='https://www.era.com.sg/salespersons/leadership-team/'
# fetch the content from url
page_response = requests.get(page_link, timeout=5)


# In[23]:

# parse html
page_content = BeautifulSoup(page_response.content, "html.parser")

# extract all html elements where price is stored
prices = page_content.find_all(class_='email-address')#[-1].extract()
prices_str = str(prices)
# prices has a form:
#[<div class="main_price">Price: $66.68</div>,
# <div class="main_price">Price: $56.68</div>]

# you can also access the main_price class by specifying the tag of the class
#prices = page_content.find_all('div', attrs={'class':'main_price'})

# boot out the last `<document>`, which contains the binary data
#soup.find_all('document')[-1].extract()

#p = soup.find_all('p')
#paragraphs = []
#for x in p:
#    paragraphs.append(str(x))


# In[24]:

print (prices_str)

prices()
# In[22]:




# In[9]:

import re


# In[29]:

matching1 = re.findall(r'[\w\.-]+@[\w\.-]+', prices_str)
for i in matching1:
    print (i)


# In[34]:

matching1[1]


# In[ ]:




# In[40]:

#have to do some splicing since my website seems to have duplicates of the email...
i=0
for i in matching1:
    matching2=matching1[::2]


# In[41]:

matching2


# In[42]:

text_file = open("ERA_emails_clean.txt", "w")
text_file.write("Email: %s" % matching2)
text_file.close()


# In[47]:

#conver the notebook to python .py script
get_ipython().system("jupyter nbconvert --to script 'scraping a website for email addresses..ipynb'")


# In[ ]:



