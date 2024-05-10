import pandas as pd
excel_file_path='predict.csv'
df=pd.read_csv(excel_file_path)

for i in range(10140):
     value_ij=df.iloc[i-1,1]
     if value_ij>0.5:
        df.iloc[i-1,1]=1
     else:
        df.iloc[i-1,1]=0

df.to_csv('result_cnn_bigramchar.csv',index=False)

