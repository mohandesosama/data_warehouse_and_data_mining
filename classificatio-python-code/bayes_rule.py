import pandas as pd
import numpy as np

def calc_class_probs(df):
    df1_decision=df.iloc[:,-1]
    # get counts of total yess and nos
    total_yes_no=df1_decision.value_counts()
    # calculate info of the entire table
    total_yes_no_index=total_yes_no.index.values
    ps={}
    counter=0
    # calculate the probabilitis of yes and no classes
    for dex in total_yes_no_index:
        ps[dex]=total_yes_no[counter]/df.shape[0]
        counter += 1
    return ps

def calc_field_probs(df,field_name,field_value):
    total_yes_no=df.iloc[:,-1].value_counts()
    dex_total=total_yes_no.index.values
    df_sub=df.query(field_name + '=="' + field_value + '"')
    field_yes_no=df_sub.iloc[:,-1].value_counts()
    dex_field=field_yes_no.index.values
    # the order total_yes_no is  ['yes' 'no'], however the order in field_yes_no is ['no' 'yes']
    probs={}
    counter=0
    # we start by 'yes' in total_yes_no and search for its corresponding 'yes' entry in field_yes_no
    dex_total_list=list(dex_total)
    dex_field_list=list(dex_field)
    for dex in dex_total_list:
        try:
            # if the class exists in the field class values
            i=dex_field_list.index(dex)
            p=field_yes_no[i]/total_yes_no[counter]
        except ValueError:
            #it should be zero, but we added small value as 'Laplacian correction'
            p=0.001/total_yes_no[counter]
        counter += 1
        probs[dex]=p
    return probs

def user_input(df):
    user_in={}
    # remove the last column, it is the class label
    df_=df.drop(df.columns[[-1]],axis=1)
    # get all the column names for the database table (excel table)
    for col in df_.columns:
        col_counts = df_[col].value_counts()
        field_value=''
        while field_value not in col_counts.index.values:
            field_value=input("Enter one of the following values {} for the field {} : ".format(col_counts.index.values,col))
        user_in[col]=field_value
    return user_in

def train_and_test(df_train_XY,df_test_X):
    #print(df_test_Y)
    class_ps=calc_class_probs(df_train_XY)
    #us_in=user_input(df)
    # get all probabilities of the fields in one list
    o=[]
    for i in range(df_test_X.shape[0]):
        # navigate the test set record by record
        record=df_test_X.iloc[i].to_dict()
        #print(record)
        ps=[calc_field_probs(df_train_XY,col,record[col]) for col in record.keys()]
        # add the prob of yes, no classes
        ps.append(class_ps)
        total_ps={}
        # multiply filed probs and class probs of yes and no and get the resuls
        for dic in ps:
            for key in dic:
                total_ps[key] = total_ps.get(key,1)* dic[key]
        # return the ouptut as 1 for yes and 0 for no
        o.append((total_ps['yes'] > total_ps['no'])*1)
        print(record)
        if total_ps['yes'] > total_ps['no']:
            print('buys_computer=yes')
        else:
            print('buys_computer=no')
    return o

if __name__=="__main__":
    df=pd.read_csv('StudentsDB.csv')
    df_train_XY = df.iloc[:14]
    df_test = df.iloc[14:]
    # we have to remove the last column from the test set
    # our algorithm will geuess it 
    df_test_X=df_test.drop(df_test.columns[[-1]],axis=1)
    df_test_Y=[(t=='yes')*1 for t in list(df_test.iloc[:,-1])]
    
    o=train_and_test(df_train_XY,df_test_X)
    
    print('Accuracy : ',np.mean([np.array(o) == np.array(df_test_Y)])*100) 
        

   


    