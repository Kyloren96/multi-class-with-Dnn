import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#导入数据
names_card = [ '学生id','消费类别','消费地点','消费方式','消费时间','消费金额','剩余金额']
path = '/Users/dyhfr/PycharmProjects/MyProject/DNN/'
card_train = pd.read_csv(path+"train/card_train.txt",header=None,encoding='utf-8',names = names_card)
card_test = pd.read_csv(path+"test/card_test.txt",header=None,encoding='utf-8',names = names_card)
card_data = pd.concat([card_train,card_test])

del card_test,card_train

name_score = ['学生id','学院编号','成绩排名']
score_train = pd.read_csv(path+"train/score_train.txt",header=None,encoding='utf-8',names = name_score)
score_test = pd.read_csv(path+"test/score_test.txt",header=None,encoding='utf-8',names = name_score)
score_data = pd.concat([score_train,score_test])

#对消费方式进行处理
card_data['消费方式'] = card_data['消费方式'].astype('category')
card_data['消费方式'].describe()

#处理缺失项
card_data.isnull().sum()
card_data['消费方式'].fillna('食堂',inplace=True)
card_sum_by_ID = card_data.groupby(['学生id'])['消费金额'].sum()
card_sum_by_ID_type = card_data.groupby(['学生id','消费方式'])['消费金额'].sum().unstack('消费方式')
card_sum_by_ID_type.fillna(0,inplace=True)
card = pd.concat([card_sum_by_ID,card_sum_by_ID_type],axis =1)
del card_sum_by_ID,card_sum_by_ID_type,card_data

#将成绩进行标准化处理
score_data.成绩排名= score_data.groupby('学院编号').成绩排名.transform(lambda x: (x - x.mean()) / x.std())
score_data.set_index('学生id')

#save data
score_data.to_csv(path+'cleaned_score.csv')
card.to_csv(path+'cleaned_card.csv')

card = pd.read_csv(path+"cleaned_card.csv",index_col='学生id')
score = pd.read_csv(path+"cleaned_score.csv",index_col='学生id')
data = pd.concat([score,card],axis =1).drop('学院编号', 1)

ID_test = pd.read_csv(path+"test/studentID_test.csv",encoding='gb18030')
ID_sub_train = pd.read_csv(path+"train/subsidy_train.csv",encoding='gb18030')

X_test = data.loc[ID_test.学生id,:]
X_train = data.loc[ID_sub_train.学生id,:]
y_train = ID_sub_train.助学金金额
X_test.to_csv(path+'X_test.csv')
X_train.to_csv(path+'X_train.csv')
y_train.to_csv(path+'y_train.csv')

X_test.to_pickle(path+'X_test.pkl')
X_train.to_pickle(path+'X_train.pkl')
y_train.to_pickle(path+'y_train.pkl')