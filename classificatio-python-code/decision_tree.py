import pandas as pd
import numpy as np

def calc_field_gain(df, field_name):
    #CALC info_D
    info_D=calc_total_info(df)
    #extract the column you want to calculate entropy for 
    df1_column = df[field_name]
    #extract counts of the values in that column
    data_vals=df1_column.value_counts()
    # extract the name of each value count, categorical values
    index_vals=data_vals.index.values
  
    info_field=0
    split_info_field=0
    formula = 'Info_'+ field_name +'(D)= '
    for i in range(len(index_vals)):
        # get data table of the first categorical value, data table for sepcific categorical value
        df_sub=df.query(field_name + '=="' + index_vals[i] + '"')
        # get last column for the above sub data table
        df_sub_last_column=df_sub.iloc[:,-1]
        # count how manyes yess and nos in the above sub data table
        vals=df_sub_last_column.value_counts()
        # split info is used to calculate gain ratio
        split_info_field -= df_sub.count()[0]/df.count()[0]* np.log2(df_sub.count()[0]/df.count()[0])
        # if the nubmer of yess is not zeor or the number of nos is not zero
        if(len(vals)>1):
            info_field += (df_sub.count()[0]/df.count()[0]*info(list(vals)))
            formula += str(df_sub.count()[0]) + '/' + str(df.count()[0]) + '*' + "I(" + str(list(vals)).replace('[','').replace(']','') + ")"
        else:
            # if yes count or no count is 0 then log2(1) will give 0, so info result is always zero
            info_field += 0
        # don't add + at the end of the formula
        if i != len(index_vals)-1:
            formula += " + "
        else: 
            formula += ' = '+str(info_field)
    print(formula)
    print('gain ratio', (info_D-info_field)/split_info_field)

    return info_D-info_field

def info(val_lst):
    tot = sum(val_lst)
    tot_info=0
    for x in val_lst:
        tot_info -= x/tot*np.log2(x/tot)
    return tot_info

def calc_total_info(df):
    #CALC info_D
    # extract the last column, the decision column
    df1_decision=df.iloc[:,-1]
    # get counts of total yess and nos
    total_yes_no=df1_decision.value_counts()
    # calculate info of the entire table
    info_D=info(list(total_yes_no))
    print('Info(D): ' + str(list(total_yes_no)), info_D)
    return info_D

if __name__ == '__main__':
    df=pd.read_csv('StudentsDB.csv')
    field_name='income'
    print('info gain of ' + field_name + ':',calc_field_gain(df,field_name))