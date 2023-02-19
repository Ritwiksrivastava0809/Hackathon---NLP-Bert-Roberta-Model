import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
df = pd.read_csv('C:\c++\My codes\python\datasolve\data\train.csv')
# print(df.head())
#  remove stopwords from eng language  and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"''",'``','-','--','..',"'s",'“','”','’','–','//','=','+'])

dates = set(['january','february','march','april','may','june','july','august','september','october','november','december','monday','tuesday','wednesday','thursday','friday','saturday','sunday'])


# tokenize the sentences and remove stopwords and punctuation and dates and numbers
def tokenize(text):
    text = text.lower()
    # remove numeric values
    text = ''.join([i for i in text if not i.isdigit()])
    tokens = word_tokenize(text)
    # remove float and int from tokens
    tokens = [token for token in tokens if not token.isnumeric()]
    tokens = [i for i in tokens if i not in stop_words]
    tokens = [i for i in tokens if i not in punctuation]
    tokens = [i for i in tokens if i not in dates]
    # serach
    return tokens

#  create a new column in the dataframe with the tokens
df['tokens'] = df['CONTEXT'].apply(tokenize)
print(df.tokens[0])
print(df.tokens[1])
print(df.tokens[2])
print(df.tokens[3])

# save the dataframe to csv file
df.to_csv('train_tokens.csv', index=False)


