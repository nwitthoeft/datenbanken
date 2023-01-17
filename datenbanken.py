import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

df1 = pd.read_excel('https://figshare.com/ndownloader/files/2364328')
df2 = pd.read_excel('https://figshare.com/ndownloader/files/2364329')
df3 = pd.read_excel('https://figshare.com/ndownloader/files/2364330')

column_sample_type = '<b> Sample</b>'
column_Ra_226 = "<b>Radioactivity concentration (Bq kg<sup>&#8211;1</sup>)</b>"
column_Th_232 = "Unnamed: 2"
column_K_40 = "Unnamed: 3"

new_df_cement_sample_coloumn_name = "Cement Sample"
new_df_fly_ash_sample_coloumn_name = "Fly ash Sample"
new_df_Ra_226_value_coloumn_name = "Ra 226 Value"
new_df_Ra_226_std_div_coloumn_name = "Ra 226 standard deviation"
new_df_Th_232_value_coloumn_name = "Th 232 Value"
new_df_Th_232_std_div_coloumn_name = "Th 232 standard deviation"
new_df_K_40_value_coloumn_name = "K 40 Value"
new_df_K_40_std_div_coloumn_name = "K 40 standard deviation"

column_name_arr = ["Cement Sample", "Fly ash Sample", "Ra 226 Value", "Ra 226 standard deviation",
                   "Th 232 Value", "Th 232 standard deviation", "K 40 Value", "K 40 standard deviation"]


def data_filter(data, column_index_string):
    # Write your solution here and remove pass
    str1 = ""
    str2 = ""
    data1 = []
    for element in data.index:
        str1 = data.loc[element, column_index_string]
        for i in range(len(str1)-1):
            if (str1[i] == "±"):
                break
            str2 += str1[i]
        data1.append(float(str2))
        str2 = ""
    return (data1)


def error_filter(data, column_index_string):
    # Write your solution here and remove pass
    str1 = ""
    str2 = ""
    error1 = []
    for element in data.index:
        str1 = data.loc[element, column_index_string]
        for i in range(len(str1)-1, 0, -1):  # reverse
            if (str1[i] == "±"):
                break
            str2 += str1[i]
        error1.append(float(str2[::-1]))
        str2 = ""
    return (error1)


'''
def create_dataframe(column_type_name, row_type_arr, Ra_226_column_name_val_arr, Ra_226_val_arr, Ra_226_column_name_std_dev_arr, Ra_226_std_dev_arr):
    # np_arr = np.asarray(val_arr)
    # df = pd.DataFrame(np_arr,
    # columns=[column_name_val_arr])
    # initialize data of lists.
    data = {column_type_name: row_type_arr,
            Ra_226_column_name_val_arr: Ra_226_val_arr, Ra_226_column_name_std_dev_arr: Ra_226_std_dev_arr}
    new_df = pd.DataFrame(data)
    new_df = new_df.reset_index(drop=True)
    display(new_df)
    return ()
'''


def create_dataframe1(column_name_arr, row_type_arr, Ra_226_val_arr, Ra_226_std_dev_arr, Th_232_val_arr, Th_232_std_dev_arr, K_40_val_arr, K_40_std_dev_arr):
    # np_arr = np.asarray(val_arr)
    # df = pd.DataFrame(np_arr,
    # columns=[column_name_val_arr])
    # initialize data of lists.
    data = {column_name_arr[0]: row_type_arr, column_name_arr[2]: Ra_226_val_arr, column_name_arr[3]: Ra_226_std_dev_arr, column_name_arr[4]: Th_232_val_arr, column_name_arr[5]: Th_232_std_dev_arr, column_name_arr[6]: K_40_val_arr, column_name_arr[7]: K_40_std_dev_arr}
    new_df = pd.DataFrame(data)
    new_df = new_df.reset_index(drop=True)
    display(new_df)
    return ()


error1_vektor_cement_Ra_226 = error_filter(df1[2:9], column_Ra_226)
error2_vektor_cement_Th_232 = error_filter(df1[2:9], column_Th_232)
error3_vektor_cement_K_40 = error_filter(df1[2:9], column_K_40)
# print(best_ranking(df1[2:9]))
# chart = sns.barplot(data = df1, x= df1[2:9]['<b> Sample</b>'],  y = data_filter(df1[2:9]), errorbar=None)

x_cement = df1[2:9][column_sample_type]
y_cement_Ra_226 = data_filter(df1[2:9], column_Ra_226)
y_cement_Th_232 = data_filter(df1[2:9], column_Th_232)
y_cement_K_40 = data_filter(df1[2:9], column_K_40)

fig, ax = plt.subplots(3)
ax[0].bar(x_cement, y_cement_Ra_226, yerr=error1_vektor_cement_Ra_226,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylabel("Ra 226 in Bq/kg")
ax[0].set_xlabel("cement type")
ax[0].set_title('Radioactive concetration of Ra 226 in different cement types')

ax[1].bar(x_cement, y_cement_Th_232, yerr=error2_vektor_cement_Th_232,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylabel("Th 232 in Bq/kg")
ax[1].set_xlabel("cement type")
ax[1].set_title('Radioactive concetration of Th 232 in different cement types')

ax[2].bar(x_cement, y_cement_K_40, yerr=error3_vektor_cement_K_40,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[2].set_ylabel("K 40 in Bq/kg")
ax[2].set_xlabel("cement type")
ax[2].set_title('Radioactive concetration of K 40 in different cement types')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                    top=0.9, wspace=0.4, hspace=0.4)
fig.set_size_inches(8, 16)
fig.suptitle('cement')
plt.savefig('figure.png', dpi=300)


error1_vektor_fly_ash_Ra_226 = error_filter(df1[11:16], column_Ra_226)
error2_vektor_fly_ash_Th_232 = error_filter(df1[11:16], column_Th_232)
error3_vektor_fly_ash_K_40 = error_filter(df1[11:16], column_K_40)
# print(best_ranking(df1[2:9]))
# chart = sns.barplot(data = df1, x= df1[2:9]['<b> Sample</b>'],  y = data_filter(df1[2:9]), errorbar=None)

x_fly_ash = df1[11:16][column_sample_type]
y_fly_ash_Ra_226 = data_filter(df1[11:16], column_Ra_226)
y_fly_ash_Th_232 = data_filter(df1[11:16], column_Th_232)
y_fly_ash_K_40 = data_filter(df1[11:16], column_K_40)

fig, ax = plt.subplots(3)
ax[0].bar(x_fly_ash, y_fly_ash_Ra_226, yerr=error1_vektor_fly_ash_Ra_226,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylabel("Ra 226 in Bq/kg")
ax[0].set_xlabel("fly ash type")
ax[0].set_title(
    'Radioactive concetration of Ra 226 in different fly ash types')

ax[1].bar(x_fly_ash, y_fly_ash_Th_232, yerr=error2_vektor_fly_ash_Th_232,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylabel("Th 232 in Bq/kg")
ax[1].set_xlabel("fly ash type")
ax[1].set_title(
    'Radioactive concetration of Th 232 in different fly ash types')

ax[2].bar(x_fly_ash, y_fly_ash_K_40, yerr=error3_vektor_fly_ash_K_40,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[2].set_ylabel("K 40 in Bq/kg")
ax[2].set_xlabel("fly ash type")
ax[2].set_title('Radioactive concetration of K 40 in different fly ash types')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                    top=0.9, wspace=0.4, hspace=0.4)
fig.set_size_inches(8, 16)
fig.suptitle('fly ash')
plt.savefig('figure1.png', dpi=300)
# create_dataframe(new_df_cement_sample_coloumn_name, x_cement, new_df_Ra_226_value_coloumn_name, y_cement_Ra_226, new_df_Ra_226_std_div_coloumn_name, error1_vektor_cement_Ra_226)
create_dataframe1(column_name_arr, x_cement,  y_cement_Ra_226, error1_vektor_cement_Ra_226,
                  y_cement_Th_232, error2_vektor_cement_Th_232, y_cement_K_40, error3_vektor_cement_K_40)
plt.show()
