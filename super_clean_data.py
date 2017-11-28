import pandas as pd

b_dir = '../used-cars-database/'

df = pd.read_csv(b_dir + 'autos.csv', encoding='latin1')

df['nrOfPictures'].sum()
work_data = df.drop('nrOfPictures', axis=1)
work_data.groupby('seller').size()
work_data = work_data[work_data.seller != 'gewerblich']
work_data = work_data.drop('seller', axis=1)

work_data.groupby('offerType').size()
work_data = work_data[work_data.offerType != 'Gesuch']
work_data = work_data.drop('offerType', 1)

len(work_data.groupby('name').size())
work_data = work_data.drop('name', 1)

work_data = work_data.drop('abtest', 1)

work_data = work_data[work_data.price < 100000]

len(work_data[work_data.price == 0])

work_data = work_data[work_data.price != 0]

work_data = work_data[(work_data.yearOfRegistration >= 1863)
                      & (work_data.yearOfRegistration < 2017)]

work_data = work_data[(work_data.powerPS > 0) & (work_data.powerPS < 1000)]

superclean_data = work_data.dropna()
# https://www.kaggle.com/uodasuodas/cleaned-used-cars-database

superclean_data.to_csv('superclean_data.csv')
