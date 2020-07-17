import pandas as pd

df1_fake = pd.read_csv("Fake.csv")
df1_fake['label'] = 0 #Fake news
df1_true = pd.read_csv("True.csv")
df1_true['label'] = 1 #True news

#Combining the fake and true data
data = pd.concat([df1_fake.sample(n=7000), df1_true.sample(n=7100)])

#Make label into categorical data
data['label'] = data['label'].astype('category')

#Combine the text data into one column
data['combined_text'] = data['title'] + '. ' +  data['text']

data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)
data = data.reset_index(drop=True)


df2 = pd.read_csv('train.csv')
df2['combined_text'] = df2['title'] + '. ' + df2['text']
df2 = df2.drop(df2.columns[[0, 1, 2, 3]], axis=1)
df2.dropna(subset=['combined_text'],axis=0,inplace=True)
df2.loc[df2['label'] == 0, 'label'] = -1
df2.loc[df2['label'] == 1, 'label'] = 0
df2.loc[df2['label'] == -1, 'label'] = 1

data = pd.concat([data, df2.sample(n=14000)])
data.head()


df3 = pd.read_csv('fake2.csv')
df3['combined_text'] = df3['title'] + '. ' + df3['text']
df3['label'] = 0
df3['label'] = df3['label'].astype('category')
df3.dropna(subset=['combined_text'],axis=0,inplace=True)