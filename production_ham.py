#pull raw ham emails
#Pull 1 month HAM Data
import email, getpass, imaplib, os, re, sys
import BeautifulSoup as bs
import nltk
from email.header import decode_header, make_header
import base64
from email.Parser import Parser as EmailParser
from email.utils import parseaddr
from StringIO import StringIO
from pandas import DataFrame, Series
from bs4 import BeautifulSoup
from collections import defaultdict
import email.Parser
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


user = raw_input("Enter your Gmail username ->")
pwd = getpass.getpass("Enter password ->")


m = imaplib.IMAP4_SSL("imap.gmail.com")
m.login('user', 'pwd')
m.select('"[Gmail]/All Mail"')

#result, data = m.uid('search', None, "ALL")
result, data = m.uid('search', None, '(SINCE 12-December-2015)')

#remove uncessary spaces
def removeinvalid(y):
    y=''.join([x for x in y if ord(x) < 128])
    return y

if result == 'OK': 
    for num in data[0].split():
        result, data = m.uid('fetch', num, '(RFC822)')
        full_message = ""
        msg_dict = None
        
        body_base64 = ""
        if result == 'OK':
            raw_email = data[0][1]
            
            email_message = email.message_from_string(raw_email)
            
            
            raw_email = data[0][1] # here's the body, which is raw text of the whole email
            # including headers and alternate payloads
            msg = email.message_from_string(raw_email)
            
            #msg["Message-ID"] = str(msg['Message-ID'])
            msg["Subject"] = str(decode_header(msg['Subject']))
            
            #msg_header = msg.items()
            #print msg.items()
            
            #key = msg["Message-ID"]
            
            msg_dict = msg.items()
            
            msg_dict = dict(msg_dict)
            
            #msg_dict = msg_dict.encode('utf-8', 'ignore')
            
            body_part = {}
            for part in msg.walk():

                body_part = {
                    #"Message-ID": key,
                    "MESSAGE_TYPE": str(part.get_content_maintype()),
                    "CHARSETS": str(part.get_charsets()),
                    "DEFULAT_TYPE:": str(part.get_default_type()),
                    "CONTENT_CHARSET:": str(part.get_content_charset())}
            
            body = {}
            text = ""
            '''
            check = ""
            parser = email.parser.Parser()
            parts = msg.get_payload()
            check = parts[0].get_content_type()
            if msg.is_multipart():
                if check == "text/plain":
                    text = parts[0].get_payload()
                
                #print parts[0].get_payload()
                elif check == "multipart/alternative":
                    part = parts[0].get_payload()
                
                #print parts[0].get_payload()
                    if part[0].get_content_type() == "text/plain":
                        try:
                            text = part[0].get_payload()
                        except:
                            pass
            '''
            if msg.is_multipart():
                  for part in msg.walk():
                            try:
                                    text = part.get_payload(decode=True)
                                    missing_padding = 4 - len(text) % 4

                                    if missing_padding:
                                        text += b'='* missing_padding

                                        text += base64.urlsafe_b64decode(text)
                                        #print body_base64 + "&"*80
                                    else:
                                        text += base64.urlsafe_b64decode(body)
                                        #print body_base64 + "&"*80

                            except:
                                pass

            else:
                if text is None:
                    charset = str(msg.get_content_charset())
                    text = unicode(msg.get_payload(decode=True),charset, "ignore").encode('utf8','ignore')
                    text = text.strip()
            '''
            #if text is not None:
                #text = text.strip()
            #else:
                #html = html.strip()

            '''
        #text = BeautifulSoup(text.decode('utf-16', 'ignore'))
        text = removeinvalid(str(text))
        soup = BeautifulSoup(text)
        text = soup.get_text()
        
        msg_dict['body_part'] = body_part
        msg_dict['body'] = text

                    
        from pymongo import MongoClient
        try:
            client = MongoClient('mongodb://192.168.67.90:27017')
            print "Connected successfully"
        except pymongo.errors.ConnectionFailure, e:
            print "Could not connect to MongoDB: %s" % e
        db = client['spam_database']
        msg_ham = db.production_ham

        all_ham_msg = msg_ham.insert(msg_dict)


        all_ham_msg

#run cleanse script for ham features
import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import re
pd.set_option('display.max_colwidth', -1)

#length of imported dataframe
length = len(dataframe_ham)

%matplotlib inline
feature_first_time_from = DataFrame(dataframe_ham.duplicated(['From']))

feature_first_time_from.columns = ['feature_first_time_from']
features_ham = pd.concat([dataframe_ham, feature_first_time_from], axis=1)


#Chart of First Time Sender
#first_time_sender_counts = DataFrame(features_ham.feature_first_time_from.feature_first_time_from)
#first_time_counter = feature_first_time_from.feature_first_time_from.value_counts()
#first_time_counter.plot(kind="bar")

#Links in body
bodylink = dataframe_ham['body']
#bodylink = str(bodylink)
url_pattern = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]{0,255}[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]?|(\([^\s()<>]?\)))?\))?(?:\(([^\s()<>]?|(\([^\s()<>]?\)))?\)|[^\s`!()\[\]{};:\'".,<>?  \xab\xbb\u201c\u201d\u2018\u2019]))'

extracted_data = bodylink.str.extract(url_pattern)
body_url = DataFrame(extracted_data[0])

#counts of body_url links
body_url.columns = ['feature_body_url']
body_url = DataFrame(body_url['feature_body_url'])
body_url['feature_body_url'].value_counts()

#Subject Charset Type:
subject = dataframe_ham['Subject']
subject = DataFrame(subject)
subject['Subject_raw'], subject['Subject_content_type'] = zip(*subject['Subject'].apply(lambda x: x.rsplit(', ', 1)))
subject['Subject_content_type'] = subject['Subject_content_type'].map(lambda x: x.rstrip(')]'))
subject['Subject_raw'] = subject['Subject_raw'].map(lambda x: x.lstrip('[('))

#Subject content Type
subject_content_type = DataFrame(subject['Subject_content_type'])
subject_content_type = subject_content_type.reset_index()
features_ham = pd.concat([features_ham, subject_content_type], axis=1)


#ContentType_body
ContentType_df = DataFrame(dataframe_ham['Content-Type'])
ContentType_df = ContentType_df['Content-Type'].str.split(';', return_type='frame')
#ContentType_df
ContentType_df.columns = ['ContentType_body','charset', 'extra1', 'extra2']
#ContentType_df.columns = ['ContentType_body','charset', 'extra1', 'extra2', 'extra3']
#ContentType_df['ContentType_body'].value_counts() 
ContentType_body = DataFrame(ContentType_df)
features_host = ContentType_body.reset_index()
features_ham = pd.concat([features_ham, ContentType_body], axis=1)

#raw text from body 
#Body text
#Body text
import sys  

#reload(sys)  
#sys.setdefaultencoding('utf8')


import urllib
from bs4 import BeautifulSoup

df_html_to_text = {}

def html_to_text(soup, lineNo):
    global df_html_to_text
    
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text= ''.join(text.splitlines())
    
    
    
    html_to_text1 = Series(text)
    df_html_to_text[lineNo] = html_to_text1

for i in range(length):
    body_df = dataframe_ham.body[i]

    BS_test = body_df.encode('utf-16').strip()

    soup = BeautifulSoup(BS_test)
    html_to_text(soup, i)

df_html_to_text_series = Series(df_html_to_text)
#Body Text
df_html_to_text_df = DataFrame(df_html_to_text_series, columns=['body_text'])

##IP
recieved_spf = dataframe_ham['Received-SPF']
ip_pattern = r'(?<=client-ip=)([0-9\.]*)(?=;)'
ip_address = recieved_spf.str.extract(ip_pattern)

#ip_address to dataFrame
ip_of_sender = DataFrame(ip_address)
ip_of_sender = DataFrame(ip_of_sender['Received-SPF'])
ip_of_sender.columns = ['ip_of_sender']
ip_of_sender = DataFrame(ip_of_sender)
#ip_of_sender['Received-SPF'].isnull().value_counts()
ip_of_sender = ip_of_sender.reset_index()

features_ham = pd.concat([features_ham, ip_of_sender], axis=1)


#LINK DataFrame
def regex_link(text):
    link = str(re.findall(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]{0,255}[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]?|(\([^\s()<>]?\)))?\))?(?:\(([^\s()<>]?|(\([^\s()<>]?\)))?\)|[^\s`!()\[\]{};:\'".,<>?  \xab\xbb\u201c\u201d\u2018\u2019]))', text))
    link.encode('utf-8').strip()
    if len(link) == 0:
        link = "None"
    return link
    
link_df = dataframe_ham['body'].apply(regex_link)

arr = np.array(link_df)
appended_data = []

for i,x in enumerate(arr):
    if x != "None":
        
        for y in x:

            if y != "None":
                #print i,y[0]
                a = i,y[0]
                appended_data.append(a)
            else:
                #print "None"
                none = "None" 
                appended_data.append(none)
    else:
        #print i, "None"
        none2 = i,"None"
        appended_data.append(none2)


appended_data=DataFrame(appended_data)
appended_data.columns = ['column_num', 'link']
appended_data= appended_data.set_index(['column_num', 'link'])


#HOST 
import urlparse
from pandas import DataFrame
#url = 'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transi'
#o = urlparse.urlparse(url)
#o.hostname

def url_to_host(link):
    host = urlparse.urlparse(link)
    hostname = str(host.hostname)
    return hostname


reset = appended_data.reset_index()
reset_df = DataFrame(pd.pivot_table(reset,index=['column_num'],values='link',aggfunc=lambda x: ', '.join(x)))
body_link = DataFrame(reset_df['link'])
body_link_dict = body_link.to_dict()
#body full links 
body_link_df = DataFrame(body_link_dict)

body_link_df = body_link_df.reset_index()
features_ham = pd.concat([features_ham, body_link_df], axis=1)

#reset['link_host'] = reset['link_host']
reset['link_host'] = reset['link'].apply(url_to_host)

#drop duplicate links
reset_dedup_if = reset.drop_duplicates(cols=['link_host', 'column_num'], take_last=True)
#remove www.w3.org
reset_dedup_if['link_host'] = reset_dedup_if['link_host'].str.replace(r'www.w3.org', 'None')

reset_dedup_if = DataFrame(reset_dedup_if, columns=[['column_num', 'link', 'link_host']])
reset_dedup_if = reset_dedup_if.set_index(['column_num', 'link', 'link_host'])

reset_link_host_reset = reset_dedup_if.reset_index()
reset_link_host_reset = DataFrame(reset_link_host_reset, columns=[['column_num', 'link_host']])





features_host = DataFrame(pd.pivot_table(reset_link_host_reset, index=['column_num'],values='link_host',aggfunc=lambda x: ', '.join(x)))

#LINKHOST 
features_host = DataFrame(features_host)
features_host = features_host.reset_index()
features_ham = pd.concat([features_ham, features_host], axis=1)

#get suffix of links 
import tldextract
#url = 'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transi'
#o =tldextract.extract(url)
#o.suffix


def url_to_suffix(link):
    parsed_suffix = tldextract.extract(link)
    suffix = str(parsed_suffix.suffix)
    return suffix

reset['link_suffix'] = reset['link'].apply(url_to_suffix)
reset = DataFrame(reset)
#replace blanks with None
reset['link_suffix'] = reset['link_suffix'].replace(r'', 'None', regex=True)

#allignment 
reset_suffix = DataFrame(reset, columns=[['column_num', 'link', 'link_host', 'link_suffix']])
reset_suffix = reset_suffix.set_index(['column_num', 'link', 'link_host', 'link_suffix'])

reset_link_suffix_reset = reset_suffix.reset_index()

reset_link_suffix_reset = DataFrame(reset_link_suffix_reset, columns=[['column_num', 'link_suffix']])
reset_link_suffix_reset
features_suffix = DataFrame(pd.pivot_table(reset_link_suffix_reset, index=['column_num'],values='link_suffix',aggfunc=lambda x: ', '.join(x)))

#Suffix Types
features_suffix = DataFrame(features_suffix)
features_suffix = features_suffix.reset_index()
features_ham = pd.concat([features_ham, features_suffix], axis=1)
                      
#Count of number of unique links in body (counting the host)
features_host['#unique_links'] = features_host.link_host.apply(lambda x: len(x.split(',')))

unique_link_count = DataFrame(features_host['#unique_links'])
unique_link_count = unique_link_count.reset_index()
features_ham = pd.concat([features_ham, unique_link_count], axis=1)


#Count of number of links in body (counting the suffix)
features_suffix['#links'] = features_suffix.link_suffix.apply(lambda x: len(x.split(',')))
link_count = DataFrame(features_suffix['#links'])


#add features_suffix to ham_features_dataframe
link_count = link_count.reset_index()
features_ham = pd.concat([features_ham, link_count], axis=1)

#Bag of Words 
df_html_to_text_df = DataFrame(df_html_to_text_df)
df_html_to_text_df.applymap(str)
strip_list = df_html_to_text_df.body_text.tolist()
strip_list = DataFrame(strip_list)
strip_list.columns = ['body_text']


#Normalize body text
#Normalize body text
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
stop  = stopwords.words("english")

#df_html_to_text_df = DataFrame(df_html_to_text_df)
#df_html_to_text_df.applymap(str)
#df_html_to_text_df.drop('body_text_normalize', 1)



def regex_normalize(text_body):
    normal_text = re.sub("[^a-zA-Z]", " ", text_body)
    return normal_text

def regex_lower(clean_body):
    normal_text_2 = clean_body.lower().split()
    return normal_text_2

def remove_short_words(clean_me):
    text1 = re.sub(r'\b\w{1,2}\b', '',  clean_me)
    return text1

#only words
strip_list['body_text_normalize'] = strip_list.body_text.apply(regex_normalize)
#remove short words less than 3 characters
strip_list['body_text_normalize'] = strip_list.body_text_normalize.apply(remove_short_words)
#lower
strip_list['body_text_normalize'] = strip_list.body_text_normalize.apply(regex_lower)


#remove stopwords
strip_list['body_text_normalize'] = strip_list['body_text_normalize'].apply(lambda x: [item for item in x if item not in stop])
clean_train_reviews = strip_list.body_text_normalize

#Bag of Words for Body
clean_train_reviews = DataFrame(clean_train_reviews)
clean_train_reviews = clean_train_reviews.reset_index()
features_ham = pd.concat([features_ham, clean_train_reviews], axis=1)
                         

#bag of words for subject
REGEX = re.compile(r"[^a-zA-Z]")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

subject['Subject_cleanse'] = subject['Subject_raw'].apply(tokenize)
subject['Subject_cleanse'] = DataFrame(subject['Subject_cleanse'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(tokenizer=tokenize, min_df=0, charset_error="ignore", stop_words="english", max_features=200)
counts = cv.fit_transform(subject.Subject_raw.values).toarray().ravel()
words = np.array(cv.get_feature_names())
#normalize
counts = counts / float(counts.max())
#data.shape
#words
str_subject_words = str(subject['Subject_cleanse'])

#Bag Of Words Subject
str_subject_words_df = DataFrame(subject['Subject_cleanse'])
str_subject_words_df = str_subject_words_df.reset_index()
features_ham = pd.concat([features_ham, str_subject_words_df], axis=1)


#ListUnsubscribe
features_list_unsubscribe_included = pd.notnull(features_ham['List-Unsubscribe'])
features_list_unsubscribe_included = DataFrame(features_list_unsubscribe_included)
features_list_unsubscribe_included.columns = ['ListUnsubscribe_TF']
features_list_unsubscribe_included = features_list_unsubscribe_included.reset_index()
features_ham = pd.concat([features_ham, features_list_unsubscribe_included], axis=1)

features_ham = features_ham.T.groupby(level=0).first().T


features_final = features_ham[['From', 'Subject', 'Date', 'Delivered-To', 'Received', 'Received-SPF', 'Reply-To', 'Return-Path', 'X-Received', '_id', 'feature_first_time_from', 'ListUnsubscribe_TF', 'Subject_cleanse', 'ContentType_body', 'Subject_content_type', 'ip_of_sender', 'link_host', 'body', 'body_part', 'body_text_normalize', '#links', '#unique_links','link_suffix','link_host','link']]
features_final['ham/spam'] = 'ham' 
features_final


#insert features final of ham to mongo
from pymongo import MongoClient
try:
    client = MongoClient('mongodb://192.168.67.90:27017')
    print "Connected successfully"
            
            
except pymongo.errors.ConnectionFailure, e:
    print "Could not connect to MongoDB: %s" % e
            
db = client['spam_database']
msg_spam = db.ham_clean
all_spam_msg = msg_spam.insert_many(features_final.to_dict('records'))
all_spam_msg

#send feature / cleanse data to mongo
from pymongo import MongoClient
try:
    client = MongoClient('mongodb://192.168.67.90:27017')
    print "Connected successfully"
            
            
except pymongo.errors.ConnectionFailure, e:
    print "Could not connect to MongoDB: %s" % e
            
db = client['spam_database']
msg_spam = db.production_ham_features
all_spam_msg = msg_spam.insert_many(features_final.to_dict('records'))
all_spam_msg

#pull feature / cleanse data from mongo
from pandas import DataFrame
from pymongo import MongoClient
client = MongoClient('mongodb://192.168.67.90:27017')
client.database_names()
db = client['spam_database']
collection = db.production_ham_features
dataframe_ham_clean = DataFrame(list(collection.find()))
dataframe_ham_clean