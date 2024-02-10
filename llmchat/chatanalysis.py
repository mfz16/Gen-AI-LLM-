import re
import pandas as pd
conversation = "D:/test/chat/_chat.txt "
date_list=[]
time_list=[]
sender_list=[]
message_list=[]
with open(conversation, encoding="utf-8") as fp:
    while True:
        line=fp.readline()

        #print(line)
        pattern = '([0-9]{1,2}/[0-9]{1,2}/[0-9]{1,2}, [0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})'
        #pattern = '([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
        result = re.findall(pattern, line)
        #print(result)
        if result :
            splitline = line.split('] ')
            date,time=splitline[0].split(',')
            date = date[1:]
            date_list.append(date)
            time = time[1:]
            time_list.append(time)
            #print("splitline 1",splitline[1])
            if re.findall(r"changed the subject|was added|added|changed this group's icon|left|changed their phone number|deleted this group's icon",splitline[1]):
                #print("found")
                sender=None
                sender_list.append(sender)
                message=None
                message_list.append(message)
            else:
                sender, message = splitline[1].split(': ',1)
                sender_list.append(sender)
                message_list.append(message)
        #else:
            #date=None
            #time=None
        #date=date[1:]
        #time=time[1:]
        #print(date)
        #print(time)
        #print(splitline[1])
        #print(sender)
        #print(message)
        if not line:
            break


df=pd.DataFrame(list(zip(date_list, time_list,sender_list,message_list)),
               columns =['Date', 'Time','Sender','Message'])
print(df.tail())
print(df['Message'])
df_short=pd.DataFrame(list(zip(sender_list,message_list)),columns =['Sender','Message'])
df_short=df_short.dropna(axis = 0, how = 'all')
#df_short.to_csv("d:/test/chat/clean_chat_short.csv")
#df_short.to_csv('d:/test/chat/chat_in_txt.txt', sep='\t', index=False)

chat_data = [f"[NAME] {name} [MESSAGE] {message}" for name, message in zip(df["Sender"].str.strip(), df["Message"].str.strip())]
mydf=pd.DataFrame(chat_data)
mydf.to_csv('d:/test/chat/chat_data_tok.txt',sep='\t',index=False)
#df.to_csv("d:/test/chat/clean_chat.csv")
#print(date)
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#import os
#os.environ['TRANSFORMERS_CACHE'] = 'D:/Transformer/preTrained/cache'
#from transformers import pipeline

#generator = pipeline(task="text-generation")
#generator(
#    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
#)  # doctest: +SKIP

# import tensorflow as tf
# if tf.test.gpu_device_name():

#     print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

# else:

#    print("Please install GPU version of TF")