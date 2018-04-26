import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#导入数据
names_card = [ '学生id','消费类别','消费地点','消费方式','消费时间','消费金额','剩余金额']
path = '/Users/dyhfr/PycharmProjects/MyProject/DNN/'
card_train = pd.read_csv(path+"train/card_train.txt",header=None,encoding='utf-8',names = names_card)
card_test = pd.read_csv(path+"test/card_test.txt",header=None,encoding='utf-8',names = names_card)
card_data = pd.concat([card_train,card_test])

card_data['消费方式'] = card_data['消费方式'].astype('category')
card_data.isnull().sum()
card_data['消费方式'].fillna('食堂',inplace=True)
card_sum_by_ID = card_data.groupby(['学生id'])['消费金额'].sum()
card_sum_by_ID_type = card_data.groupby(['学生id','消费方式','消费时间'])['消费金额'].sum().unstack('消费方式')
card_sum_by_ID_type.fillna(0,inplace=True)
print(card_sum_by_ID_type.head(10))
card = pd.concat([card_sum_by_ID,card_sum_by_ID_type],axis =1)
#print(card.head(10))
