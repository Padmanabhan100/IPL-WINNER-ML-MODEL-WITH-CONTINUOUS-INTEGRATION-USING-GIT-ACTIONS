import pandas as pd

df = pd.read_csv("data/matches.csv")


na_index = df[df['city'].isna()].index
for index in na_index:
    df['city'][index] = "Dubai"

# Dropping rows with unknown winners
df.drop(df[df['winner'].isna()].index,inplace=True)

# Filling with mode
df['umpire1'].fillna("HDPK Dharmasena",inplace=True)
df['umpire2'].fillna("C Shamshuddin",inplace=True)

# Dropping umpire 3 column
del df['umpire3']

# fetaure engineering
day = []
month = []
year = []
for date in df['date']:
    date = date.split("-")
    day.append(date[0])
    month.append(date[1])
    year.append(date[2])

df['day'] = day
df['month'] = month
df['year'] = year

del df['date']
del df['Season']

# creating a copy of dataset
train = df.copy()
del train['id']

## Preparing features for onehot encoding
train['team1'] = 'team1 '+ train['team1']
train['team2'] = 'team2 ' + train['team2']
train['toss_winner'] = 'toss_winner ' + train['toss_winner']
train['winner'] = 'winner_' + train['winner']
dummy = pd.get_dummies(train[['city','team1','team2','toss_winner','venue']])
train = pd.concat([train,dummy],join='inner',axis=1)
train.drop(columns=['city','team1','team2','toss_winner','venue'],axis=1,inplace=True)


#Encoding toss_decision
mapp = {"field":1,'bat':0}

train['toss_decision'] = train['toss_decision'].map(mapp)

# Encoding Result
mapp = {'normal':0,'tie':1}
train['result'] = train['result'].map(mapp)

# encoding teams
mapp = {
        'winner_Sunrisers Hyderabad':0,
        'winner_Rising Pune Supergiant':1,
       'winner_Kolkata Knight Riders':2,
       'winner_Kings XI Punjab':3,
       'winner_Royal Challengers Bangalore':4,
       'winner_Mumbai Indians':5,
       'winner_Delhi Daredevils':6,
       'winner_Gujarat Lions':7,
       'winner_Chennai Super Kings':8,
       'winner_Rajasthan Royals':9,
       'winner_Deccan Chargers':10,
       'winner_Pune Warriors':11,
       'winner_Kochi Tuskers Kerala':12,
       'winner_Rising Pune Supergiants':13,
       'winner_Delhi Capitals':14
}

train['winner'] = train['winner'].map(mapp)

# removing useless columns
train.drop(columns=['player_of_match','umpire1','umpire2','win_by_runs','win_by_wickets'],axis=1,inplace=True)

# changing datatype
train['day'] = train['day'].astype('int64')
train['month'] = train['month'].astype('int64')
train['year'] = train['year'].astype("int64")

# resampling
from sklearn.utils import resample
train_major = train[train['winner']==5]  # MAJOR CLASS

# MINOR CLASSES 
train_minor1 = train[train['winner']==0]
train_minor2 = train[train['winner']==1]
train_minor3 = train[train['winner']==2]
train_minor4 = train[train['winner']==3]
train_minor5 = train[train['winner']==4]
train_minor6 = train[train['winner']==6]
train_minor7 = train[train['winner']==7]
train_minor8 = train[train['winner']==8]
train_minor9 = train[train['winner']==9]
train_minor10 = train[train['winner']==10]
train_minor11 = train[train['winner']==11]
train_minor12 = train[train['winner']==12]
train_minor13 = train[train['winner']==13]
train_minor14 = train[train['winner']==14]

# upsamples data
train_minor1_sampled = resample(train_minor1,n_samples=109)
train_minor2_sampled = resample(train_minor2,n_samples=109)
train_minor3_sampled = resample(train_minor3,n_samples=109)
train_minor4_sampled = resample(train_minor4,n_samples=109)
train_minor5_sampled = resample(train_minor5,n_samples=109)
train_minor6_sampled = resample(train_minor6,n_samples=109)
train_minor7_sampled = resample(train_minor7,n_samples=109)
train_minor8_sampled = resample(train_minor8,n_samples=109)
train_minor9_sampled = resample(train_minor9,n_samples=109)
train_minor10_sampled = resample(train_minor10,n_samples=109)
train_minor11_sampled = resample(train_minor11,n_samples=109)
train_minor12_sampled = resample(train_minor12,n_samples=109)
train_minor13_sampled = resample(train_minor13,n_samples=109)
train_minor14_sampled = resample(train_minor14,n_samples=109)

# Concatiate major with upsampled

train_upsampled = pd.concat([train_major,
                            train_minor1_sampled,
                            train_minor2_sampled,
                            train_minor3_sampled,
                            train_minor4_sampled,
                            train_minor5_sampled,
                            train_minor6_sampled,
                            train_minor7_sampled,
                            train_minor8_sampled,
                            train_minor9_sampled,
                            train_minor10_sampled,
                            train_minor11_sampled,
                            train_minor12_sampled,
                            train_minor13_sampled,
                            train_minor14_sampled,
                            ])

train_upsampled.to_csv("preprocessed_data/train.csv")
                            
