import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import geopandas as gpd
import matplotlib.pyplot as plt
import folium

df1 = pd.read_excel('https://figshare.com/ndownloader/files/2364328')
df2 = pd.read_excel('https://figshare.com/ndownloader/files/2364329')
df3 = pd.read_excel('https://figshare.com/ndownloader/files/2364330')

column_sample_type = '<b> Sample</b>'
column_Ra_226_df1 = '<b>Radioactivity concentration (Bq kg<sup>&#8211;1</sup>)</b>'
column_Ra_226_df3 = '<b>Radioactivity concentration (Bq kg<sup>&#8722;1</sup>)</b>'
column_Th_232 = 'Unnamed: 2'
column_K_40 = 'Unnamed: 3'

'''
new_df_cement_sample_coloumn_name = "Cement Sample"
new_df_fly_ash_sample_coloumn_name = "Fly ash Sample"
new_df_Ra_226_value_coloumn_name = "Ra 226 Value"
new_df_Ra_226_std_div_coloumn_name = "Ra 226 standard deviation"
new_df_Th_232_value_coloumn_name = "Th 232 Value"
new_df_Th_232_std_div_coloumn_name = "Th 232 standard deviation"
new_df_K_40_value_coloumn_name = "K 40 Value"
new_df_K_40_std_div_coloumn_name = "K 40 standard deviation"
'''

column_name_arr_df1 = ['Ra 226 Value', 'Ra 226 standard deviation',
                       'Th 232 Value', 'Th 232 standard deviation', 'K 40 Value', 'K 40 standard deviation']
sample_name_arr_df1 = ['Cement Sample', 'Fly ash Sample']

column_sample_type_df2 = '<b>Sample</b>'
column_sample_Ra_226_df2 = '<b>Radium equivalent activity (Bq kg<sup>&#8722;1</sup>)</b>'
column_H_ex = '<b>Hazard index</b>'
column_H_in = 'Unnamed: 3'
column_D_in = '<b>Indoor absorbed dose rate (nGy h<sup>&#8722;1</sup>)</b>'
column_E_in = '<b>Annual effective dose (mSv y<sup>&#8722;1</sup>)</b>'
colmun_I_a = '<b>Alpha index</b>'
column_I_y = '<b>Gamma index</b>'


def data_filter(data, column_index_string):
    # Write your solution here and remove pass
    str1 = ""
    str2 = ""
    data1 = []
    for element in data.index:
        str1 = data.loc[element, column_index_string]
        if (str(str1).find('-') == True):
            continue
        else:
            for i in range(len(str1)):
                if (str1[i] == "±"):
                    break
                str2 += str1[i]
            data1.append(float(str2))
            str2 = ""
    return (data1)


'''
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


def error_filter(data, column_index_string):
    # Write your solution here and remove pass
    str1 = ""
    str2 = ""
    error1 = []

    for element in data.index:
        str1 = data.loc[element, column_index_string]
        if (str(str1).find('-') != -1):
            continue
        elif (str(str1).find('±') == -1):
            error1.append(float(0))
            continue
        else:
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


def create_dataframe1(sample_name, column_name_arr, row_type_arr, Ra_226_val_arr, Ra_226_std_dev_arr, Th_232_val_arr, Th_232_std_dev_arr, K_40_val_arr, K_40_std_dev_arr):
    # np_arr = np.asarray(val_arr)
    # df = pd.DataFrame(np_arr,
    # columns=[column_name_val_arr])
    # initialize data of lists.
    data = {sample_name: row_type_arr, column_name_arr[0]: Ra_226_val_arr, column_name_arr[1]: Ra_226_std_dev_arr, column_name_arr[2]
        : Th_232_val_arr, column_name_arr[3]: Th_232_std_dev_arr, column_name_arr[4]: K_40_val_arr, column_name_arr[5]: K_40_std_dev_arr}
    new_df = pd.DataFrame(data)
    new_df = new_df.reset_index(drop=True)
    display(new_df)
    return (new_df)


def create_dataframe3(Sample, column_name_arr_df3, row_type_arr, Ra_226_val_arr, Ra_226_std_dev_arr, Th_232_val_arr, Th_232_std_dev_arr, K_40_val_arr, K_40_std_dev_arr):

    return ()


error1_vektor_cement_Ra_226 = error_filter(df1[2:9], column_Ra_226_df1)
error1_vektor_cement_Ra_226.append(
    round((sum(error1_vektor_cement_Ra_226)/len(error1_vektor_cement_Ra_226)), 1))
error2_vektor_cement_Th_232 = error_filter(df1[2:9], column_Th_232)
error2_vektor_cement_Th_232.append(
    round((sum(error2_vektor_cement_Th_232)/len(error2_vektor_cement_Th_232)), 1))
error3_vektor_cement_K_40 = error_filter(df1[2:9], column_K_40)
error3_vektor_cement_K_40.append(
    round((sum(error3_vektor_cement_K_40)/len(error3_vektor_cement_K_40)), 1))
# print(best_ranking(df1[2:9]))
# chart = sns.barplot(data = df1, x= df1[2:9]['<b> Sample</b>'],  y = data_filter(df1[2:9]), errorbar=None)
add_row = pd.Series(["AM±SD"])
x_cement = df1[2:9][column_sample_type]
x_cement = pd.concat([x_cement, add_row], ignore_index=True)

y_cement_Ra_226 = data_filter(df1[2:9], column_Ra_226_df1)
y_cement_Ra_226.append(round((sum(y_cement_Ra_226)/len(y_cement_Ra_226)), 1))
y_cement_Th_232 = data_filter(df1[2:9], column_Th_232)
y_cement_Th_232.append(round((sum(y_cement_Th_232)/len(y_cement_Th_232)), 1))
y_cement_K_40 = data_filter(df1[2:9], column_K_40)
y_cement_K_40.append(round((sum(y_cement_K_40)/len(y_cement_K_40)), 1))

cement_df = create_dataframe1(sample_name_arr_df1[0], column_name_arr_df1, x_cement,  y_cement_Ra_226, error1_vektor_cement_Ra_226,
                              y_cement_Th_232, error2_vektor_cement_Th_232, y_cement_K_40, error3_vektor_cement_K_40)

fig, ax = plt.subplots(3)
cement_Ra_226_bar = ax[0].bar(x_cement, y_cement_Ra_226, yerr=error1_vektor_cement_Ra_226,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylabel("Ra 226 in Bq/kg")
ax[0].set_xlabel("cement type")
ax[0].set_title('Radioactive concetration of Ra 226 in different cement types')

cement_Th_232_bar = ax[1].bar(x_cement, y_cement_Th_232, yerr=error2_vektor_cement_Th_232,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylabel("Th 232 in Bq/kg")
ax[1].set_xlabel("cement type")
ax[1].set_title('Radioactive concetration of Th 232 in different cement types')

cement_K_40_bar = ax[2].bar(x_cement, y_cement_K_40, yerr=error3_vektor_cement_K_40,
          align='center', alpha=0.5, ecolor='black', capsize=10)
ax[2].set_ylabel("K 40 in Bq/kg")
ax[2].set_xlabel("cement type")
ax[2].set_title('Radioactive concetration of K 40 in different cement types')

ax[0].bar_label(cement_Ra_226_bar, padding=3)
ax[1].bar_label(cement_Th_232_bar, padding=3)
ax[2].bar_label(cement_K_40_bar, padding=3)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                    top=0.9, wspace=0.4, hspace=0.4)
fig.set_size_inches(8, 16)
fig.suptitle('cement')
fig.savefig('figure.png', dpi=300)


error1_vektor_fly_ash_Ra_226 = error_filter(df1[11:16], column_Ra_226_df1)
error1_vektor_fly_ash_Ra_226.append(
    round((sum(error1_vektor_fly_ash_Ra_226)/len(error1_vektor_fly_ash_Ra_226)), 1))
error2_vektor_fly_ash_Th_232 = error_filter(df1[11:16], column_Th_232)
error2_vektor_fly_ash_Th_232.append(
    round((sum(error2_vektor_fly_ash_Th_232)/len(error2_vektor_fly_ash_Th_232)), 1))
error3_vektor_fly_ash_K_40 = error_filter(df1[11:16], column_K_40)
error3_vektor_fly_ash_K_40.append(
    round((sum(error3_vektor_fly_ash_K_40)/len(error3_vektor_fly_ash_K_40)), 1))
# print(best_ranking(df1[2:9]))
# chart = sns.barplot(data = df1, x= df1[2:9]['<b> Sample</b>'],  y = data_filter(df1[2:9]), errorbar=None)

x_fly_ash = df1[11:16][column_sample_type]
x_fly_ash = pd.concat([x_fly_ash, add_row], ignore_index=True)

y_fly_ash_Ra_226 = data_filter(df1[11:16], column_Ra_226_df1)
y_fly_ash_Ra_226.append(
    round((sum(y_fly_ash_Ra_226)/len(y_fly_ash_Ra_226)), 1))
y_fly_ash_Th_232 = data_filter(df1[11:16], column_Th_232)
y_fly_ash_Th_232.append(
    round((sum(y_fly_ash_Th_232)/len(y_fly_ash_Th_232)), 1))
y_fly_ash_K_40 = data_filter(df1[11:16], column_K_40)
y_fly_ash_K_40.append(round((sum(y_fly_ash_K_40)/len(y_fly_ash_K_40)), 1))

fly_ash_df = create_dataframe1(sample_name_arr_df1[1], column_name_arr_df1, x_fly_ash,  y_fly_ash_Ra_226, error1_vektor_fly_ash_Ra_226,
                               y_fly_ash_Th_232, error2_vektor_fly_ash_Th_232, y_fly_ash_K_40, error3_vektor_fly_ash_K_40)

fig1, ax = plt.subplots(3)
fly_ash_Ra_226_bar = ax[0].bar(x_fly_ash, y_fly_ash_Ra_226, yerr=error1_vektor_fly_ash_Ra_226,
                                align='center', alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylabel("Ra 226 in Bq/kg")
ax[0].set_xlabel("fly ash type")
ax[0].set_title(
    'Radioactive concetration of Ra 226 in different fly ash types')

fly_ash_Th_232_bar = ax[1].bar(x_fly_ash, y_fly_ash_Th_232, yerr=error2_vektor_fly_ash_Th_232,
                                align='center', alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylabel("Th 232 in Bq/kg")
ax[1].set_xlabel("fly ash type")
ax[1].set_title(
    'Radioactive concetration of Th 232 in different fly ash types')

fly_ash_K_40_bar = ax[2].bar(x_fly_ash, y_fly_ash_K_40, yerr=error3_vektor_fly_ash_K_40,
                              align='center', alpha=0.5, ecolor='black', capsize=10)
ax[2].set_ylabel("K 40 in Bq/kg")
ax[2].set_xlabel("fly ash type")
ax[2].set_title('Radioactive concetration of K 40 in different fly ash types')

ax[0].bar_label(fly_ash_Ra_226_bar, padding=3)
ax[1].bar_label(fly_ash_Th_232_bar, padding=3)
ax[2].bar_label(fly_ash_K_40_bar, padding=3)

fig1.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                     top=0.9, wspace=0.4, hspace=0.4)
fig1.set_size_inches(8, 16)
fig1.suptitle('fly ash')
fig.savefig('figure1.png', dpi=300)

# create_dataframe(new_df_cement_sample_coloumn_name, x_cement, new_df_Ra_226_value_coloumn_name, y_cement_Ra_226, new_df_Ra_226_std_div_coloumn_name, error1_vektor_cement_Ra_226)
pie_fig, ax = plt.subplots(1)
ax = plt.pie(cement_df.loc[7, ["Ra 226 Value", "Th 232 Value", "K 40 Value"]], labels=[
             'Ra_226', 'Th_232', 'K_40'])
pie_fig.savefig('figure2.png', dpi=300)
# df2
# cement
y_cement_H_ex = data_filter(df2[2:10], column_H_ex)
y_cement_H_ex_error = error_filter(df2[2:10], column_H_ex)
y_cement_H_in = data_filter(df2[2:10], column_H_in)
y_cement_H_in_error = error_filter(df2[2:10], column_H_in)

threshold_fig_cement, ax = plt.subplots(1)

threshold = 1

x_cement_label = np.arange(len(x_cement))

y_cement_H_ex = np.asarray(y_cement_H_ex)
y_cement_H_in = np.asarray(y_cement_H_in)

a_threshold_Hex = np.maximum(y_cement_H_ex - threshold, 0)
b_threshold_Hex = np.minimum(y_cement_H_ex, threshold)

a_threshold_Hin = np.maximum(y_cement_H_in - threshold, 0)
b_threshold_Hin = np.minimum(y_cement_H_in, threshold)

width = 0.35


cement_bar_1 = ax.bar(x_cement_label - width/2, b_threshold_Hex,
                      width,  color='lime', align='center', alpha=0.5, label='Hex')
cement_bar_2 = ax.bar(x_cement_label - width/2, a_threshold_Hex, width, yerr=y_cement_H_ex_error,
                      color='orange', align='center', alpha=0.5, capsize=10, bottom=b_threshold_Hex)
cement_bar_3 = ax.bar(x_cement_label + width/2, b_threshold_Hin,
                      width,  color='green', align='center', alpha=0.5, label='Hin')
cement_bar_4 = ax.bar(x_cement_label + width/2, a_threshold_Hin, width, yerr=y_cement_H_in_error,
                      color='red', align='center', alpha=0.5, capsize=10, bottom=b_threshold_Hin)

plt.axhline(threshold, color='black', ls='solid')

ax.set_xticks(x_cement_label, x_cement)
ax.legend()

# ax.bar_label(fly_ash_bar_1, padding=3)
ax.bar_label(cement_bar_2, padding=3)
# ax.bar_label(fly_ash_bar_3, padding=3)
ax.bar_label(cement_bar_4, padding=3)
plt.tight_layout()

threshold_fig_cement.savefig("figure3")

# fly ash
y_fly_ash_H_ex = data_filter(df2[11:17], column_H_ex)
y_fly_ash_H_ex_error = error_filter(df2[11:17], column_H_ex)
y_fly_ash_H_in = data_filter(df2[11:17], column_H_in)
y_fly_ash_H_in_error = error_filter(df2[11:17], column_H_in)

threshold_fig_fly_ash, ax = plt.subplots(1)

threshold = 1

x_fly_ash_label = np.arange(len(x_fly_ash))

y_fly_ash_H_ex = np.asarray(y_fly_ash_H_ex)
y_fly_ash_H_in = np.asarray(y_fly_ash_H_in)

a_threshold_Hex = np.maximum(y_fly_ash_H_ex - threshold, 0)
b_threshold_Hex = np.minimum(y_fly_ash_H_ex, threshold)

a_threshold_Hin = np.maximum(y_fly_ash_H_in - threshold, 0)
b_threshold_Hin = np.minimum(y_fly_ash_H_in, threshold)

width = 0.35
# rects1 = ax.bar(x_cement - width/2, a_threshold_Hex, width, )
# rects2 = ax.bar(x_cement + width/2, a_threshold_Hin, width, label='Hex')


fly_ash_bar_1 = ax.bar(x_fly_ash_label - width/2, b_threshold_Hex,
                       width, color='lime', align='center', alpha=0.5, label='Hex')
fly_ash_bar_2 = ax.bar(x_fly_ash_label - width/2, a_threshold_Hex, width, yerr=y_fly_ash_H_ex_error,
                       color='orange', alpha=0.5,  ecolor='black', capsize=10, bottom=b_threshold_Hex)
fly_ash_bar_3 = ax.bar(x_fly_ash_label + width/2, b_threshold_Hin,
                       width, color='green', align='center', alpha=0.5,  label='Hin')
fly_ash_bar_4 = ax.bar(x_fly_ash_label + width/2, a_threshold_Hin, width, yerr=y_fly_ash_H_in_error,
                       color='red', alpha=0.5, ecolor='black', capsize=10, bottom=b_threshold_Hin)

plt.axhline(threshold, color='black', ls='solid')

ax.set_xticks(x_fly_ash_label, x_fly_ash)
plt.legend()

# ax.bar_label(fly_ash_bar_1, padding=3)
ax.bar_label(fly_ash_bar_2, padding=3)
# ax.bar_label(fly_ash_bar_3, padding=3)
ax.bar_label(fly_ash_bar_4, padding=3)

threshold_fig_fly_ash.tight_layout()
threshold_fig_fly_ash.savefig("figure4")
# df3

plt.show()
