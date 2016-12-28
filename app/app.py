#!flask/bin/python
from flask import Flask,jsonify
import matplotlib
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
# from nltk.book import *
from nltk.tokenize import RegexpTokenizer

app = Flask(__name__)

date=np.empty((0))
time=np.empty((0))
name=np.empty((0))
messages=np.dtype(str)
messages=np.empty((0))

sender_messages=np.eye(2)

count=0
letter=0
digit=0
i=0
with open('Habibi.txt') as f:
    lines = f.readlines()
for line in lines:
	if(count==0):
		count+=1
		continue
	if(line[0].isdigit()):
		count+=1
		line=line.strip('\n')
		line=unicode(line,'utf-8')
		line=line.encode('unicode-escape')
		new_line=line
		new_line=new_line.replace(', ','-|-',1)
		new_line=new_line.replace(' - ','-|-',1)
		new_line=new_line.replace(': ','-|-',1)
		new_line=new_line.split('-|-')
		date=np.append(date,new_line[0])
		time=np.append(time,new_line[1])
		name=np.append(name,new_line[2])
		messages=np.append(messages,new_line[3])
	else:
		letter+=1
		count+=1
		line=unicode(line,'utf-8')
		line=line.encode('unicode-escape')
		messages[messages.shape[0]-1]=messages[messages.shape[0]-1]+line 
f.close()

data = {
		'date' : pd.Series(date),
     	'time' : pd.Series(time),
     	'sender': pd.Series(name),
     	'messages':pd.Series(messages)
     	}
df = pd.DataFrame(data)

tokenizer = RegexpTokenizer(r'\w+')
df['msg_length']=pd.Series(0, df.index)
df['no_of_words']=pd.Series(0, df.index)
for index,msg in enumerate(df.messages):
    tokens=tokenizer.tokenize(msg)
    df.set_value(index,'msg_length',len(msg))
    df.set_value(index,'no_of_words',len(tokens))

participants = df.sender.unique()



def no_of_messages(array):
    return array.shape[0]

def frequency(entity):
    freq=entity.value_counts(sort=True).sort_values(ascending=False)
    freq=freq.to_dict()
    result={
            'entity':pd.Series(freq.keys()),
            'freq':pd.Series(freq.values())
    }
    result= pd.DataFrame(result)
    result= result.sort_values('freq', ascending=0)
    return result.reset_index()

def group_by_date(dataFrame,date):
    gb=dataFrame.groupby('date',as_index=False)
    return gb.get_group(date)

def group_by_sender(dataFrame,sender):  
    gb=dataFrame.groupby('sender')
    return gb.get_group(sender)

def group_by_time(dataFrame,time):
    gb=dataFrame.groupby('time',as_index=False)
    return gb.get_group(time)

def message_tokens(messages):
    tokenizer = RegexpTokenizer(r'\w+')
    bag_of_words=[]
    for msg in messages:
        tokens=tokenizer.tokenize(msg)
        for word in tokens:
            bag_of_words.append(word)
    bag_of_words=pd.Series(bag_of_words)
    return bag_of_words

#most frequently used words

tokens=message_tokens(df.messages)

#frequency of most frequently used words
token_frequency=frequency(tokens)

def get_sender_token_freq(df,sender): 
    group_by_sender_data = group_by_sender(df,sender)
    sender_tokens=message_tokens(group_by_sender_data.messages)
    return frequency(sender_tokens)


def get_emoji(source):
    emoji={}
    for lit in source['entity']:
        lit=unicode(lit,'utf-8')
        lit=lit.encode('unicode-escape')
        s=lit.decode('unicode-escape')
        emoticons = re.findall(ur'[U0001f600-U0001f650]',s)
        if(len(emoticons)==9 and emoticons[0]=='U' and len(lit)==9):
            smilie='\\'+lit
            smilie=unicode(smilie,'utf-8')
            smilie=smilie.decode('unicode-escape')
            emoji[smilie]=source.loc[source['entity']==lit].freq.to_dict().values()[0]
    emoji = sorted(emoji.items(), key=lambda kv: kv[1], reverse=True)
    emoji= pd.DataFrame(emoji) 
    emoji.columns=['emoji','freq']
    return emoji

def get_emoticons_data():
    total_emoji=get_emoji(token_frequency)
    s_length=len(total_emoji['emoji'])
    sender_emoji={}
    for x in participants:
        sender=get_sender_token_freq(df,x)
        sender_emoji[x]=get_emoji(sender)
        total_emoji[x]=pd.Series(0, index=total_emoji.index)
    for emoji in total_emoji['emoji']:
        for x in sender_emoji:
            for s_emo in sender_emoji[x]['emoji']:
                if(emoji==s_emo):
                    main_index= int(np.where(total_emoji['emoji']==emoji)[0])
                    sender_index=int(np.where(sender_emoji[x]['emoji']==emoji)[0])
                    total_emoji.set_value(main_index, x, sender_emoji[x]['freq'][sender_index])
                    break
            continue   
    return total_emoji

@app.route('/')
def index():
    dates=df.date[:10]
    return jsonify(dates)
@app.route('/get_participants')
def get_participants():
    participants=[]
    for participant in df.sender.unique():
        participants.append(participant)    
    result={
    	'senders':participants
    }
    response= jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/total_messages')
def get_msg_count():
    sender_total_msg={}
    sender_contribution={}
    for x in participants:
	    sender_total_msg[x]=no_of_messages(group_by_sender(df,x))
	    sender_contribution[x]="{:.2f}".format(float(no_of_messages(group_by_sender(df,x)))/float(no_of_messages(df["sender"]))*100)
    total_no_of_msg={
	    'total_msg':no_of_messages(df["sender"]),
	    'sender':{
	        'total_msg':sender_total_msg,
	        'contribution':sender_contribution
	    }
	}
    response=jsonify(total_no_of_msg)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/total_msg_stat')
def get_total_msg_stat():
    #TOTAL_MSG_LENGTH
    total_msg_len=df['msg_length'].sum()
    # print 'total message length : ',total_msg_len
    total_words=df['no_of_words'].sum()
    # print 'total words used : ',total_words
    sender_msg_len={}
    percent_msg={}
    sender_words={}
    percent_words={}
    for x in participants:
	    sender=group_by_sender(df,x)
	    sender_msg_len[x]=sender['msg_length'].sum()
	    percent_msg[x]="{:.2f}".format(float(sender_msg_len[x])/float(total_msg_len)*100)
	    sender_words[x]=sender['no_of_words'].sum()
	    percent_words[x]="{:.2f}".format(float(sender_words[x])/float(total_words)*100)
	#     print x ,': \n total msg length :',sender_msg_len,'\t percentage :',percent_msg,'\n'
	#     print x ,': \n total words :',sender_words,'\t percentage :',percent_words,'\n'
    total_msg_stat={
	    'total_msg_len':total_msg_len,
	    'total_words':total_words,
	    'sender':{
	        'msg_len':sender_msg_len,
	        'msg_percent':percent_msg,
	        'total_words':sender_words,
	        'word_percent':percent_words
	        
	    }
	}
    response=jsonify(total_msg_stat);
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/avg_msg_stat')
def get_avg_msg_stat():
    #avg_msg_len
    avg_msg_len="{:.2f}".format(df['msg_length'].mean())
    avg_words_per_msg="{:.2f}".format(df['no_of_words'].mean())
    #AVG_MSG_LEN & AVG NO_OF_WORDS PER SENDER
    sender_avg_msg_len={}
    avg_words_per_sender={}
    for x in participants:
	    sender=group_by_sender(df,x)
	    sender_avg_msg_len[x]="{:.2f}".format(sender['msg_length'].mean())
	    avg_words_per_sender[x]="{:.2f}".format(sender['no_of_words'].mean())

    avg_msg_stat={
	    'avg_words_per_msg' : avg_words_per_msg,
	    'avg_msg_len' : avg_msg_len,
	    'sender':{
	        'avg_words_per_msg' : avg_words_per_sender,
	        'avg_msg_len' : sender_avg_msg_len
	    }
    }
    response=jsonify(avg_msg_stat);
    response.headers.add('Access-Control-Allow-Origin', '*')    
    return response

@app.route('/avg_msg_per_day')
def get_avg_msg_per_day():
    #AVG_MESSAGES_PER_DAY
    dates=frequency(df.date)
    sender_msg_stat={}
    avg_msg_len_per_day={}
    avg_words_per_day={}
    avg_msg_per_day={}
    daily_msg="{:.2f}".format(dates.freq.mean())
    for x in participants:
        sender=group_by_sender(df,x)
        sender_date_freq=frequency(sender.date)
        avg_msg_len_per_day[x]="{:.2f}".format(float(sender['msg_length'].sum()/float(len(sender_date_freq)))*100)
        avg_words_per_day[x]="{:.2f}".format(float(sender['no_of_words'].sum())/float(len(sender_date_freq))*100)
        avg_msg_per_day[x]="{:.2f}".format(sender_date_freq.freq.mean())
	    
    msg_per_day={
	     'avg_msg_len_per_day':"{:.2f}".format(float(df['msg_length'].sum())/float(len(dates))*100),
	     'avg_words_per_day':"{:.2f}".format(float(df['no_of_words'].sum())/float(len(dates))*100),
	     'avg_msg_per_day':daily_msg,
	     'sender':{
	     	'avg_msg_len_per_day':avg_msg_len_per_day,
	        'avg_words_per_day':avg_words_per_day,
	        'avg_msg_per_day':avg_msg_per_day
	     }
    }
    response=jsonify(msg_per_day);
    response.headers.add('Access-Control-Allow-Origin', '*')    
    return response
@app.route('/total_emojis')
def get_total_emojis():
    #TOTAL_EMOJI
    emoticons_data=get_emoticons_data()
    total_emojis=emoticons_data['freq'].sum()
    sender_emojis={}
    sender_emoji_contribution={}
    for x in participants:
	    sender_emojis[x]=emoticons_data[x].sum()
	    sender_emoji_contribution[x]="{:.2f}".format(float(emoticons_data[x].sum())/float(total_emojis)*100)
    total_emoji={
	    'total_emojis':total_emojis,
	    'sender':{
	        'total':sender_emojis,
	        'contribution':sender_emoji_contribution
	    }
	}
    response=jsonify(total_emoji);
    response.headers.add('Access-Control-Allow-Origin', '*')    
    return response

@app.route('/distinct_emojis')
def get_distinct_emojis():
    #DISTINCT EMOJIS
    emoticons_data=get_emoticons_data()
    sender_distinct_emoji={}
    for x in participants:
	    sender_distinct_emoji[x]=np.count_nonzero(emoticons_data[x])
    distinct_emoji={
	    'total':len(emoticons_data),
	    'sender':sender_distinct_emoji
	    
	}
    response=jsonify(distinct_emoji);
    response.headers.add('Access-Control-Allow-Origin', '*')    
    return response

@app.route('/top_emojis')
def get_top_emojis():
	#MOST_USED_EMOJI
    emoticons_data=get_emoticons_data()
    sender_most_emoji={}

    for x in participants:
	    most_emoji=[]
	    for emo in sorted(emoticons_data[x])[-10:]:
	        most_emoji.append(emoticons_data[emoticons_data[x]==emo].to_dict())
	    sender_most_emoji[x]=most_emoji

    most_used_emoji={
	    'total':emoticons_data[emoticons_data.index<10].to_dict(),
	    'sender':sender_most_emoji
	    
    }
    response=jsonify(most_used_emoji);
    response.headers.add('Access-Control-Allow-Origin', '*')    
    return response
if __name__ == '__main__':
    app.run(debug=True)