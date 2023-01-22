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

# display(df3)
# df3 = df3.drop(5)
# display(df3)
# df1
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

# df2
column_name_arr_df2 = ['H ex Value', 'H ex standard deviation',
                       'H in Value', 'H in standard deviation']
sample_name_arr_df2 = ['Cement Sample', 'Fly ash Sample']

column_sample_type_df2 = '<b>Sample</b>'
column_sample_Ra_226_df2 = '<b>Radium equivalent activity (Bq kg<sup>&#8722;1</sup>)</b>'
column_H_ex = '<b>Hazard index</b>'
column_H_in = 'Unnamed: 3'
column_D_in = '<b>Indoor absorbed dose rate (nGy h<sup>&#8722;1</sup>)</b>'
column_E_in = '<b>Annual effective dose (mSv y<sup>&#8722;1</sup>)</b>'
colmun_I_a = '<b>Alpha index</b>'
column_I_y = '<b>Gamma index</b>'

# df3
column_country = '<b>Country</b>'
column_Ra_226_df3 = '<b>Radioactivity concentration (Bq kg<sup>&#8722;1</sup>)</b>'
column_Th_232_df3 = 'Unnamed: 2'
column_K_40_df3 = 'Unnamed: 3'

column_name_arr_df3 = ['Ra 226 Value', 'Ra 226 standard deviation',
                       'Th 232 Value', 'Th 232 standard deviation', 'K 40 Value', 'K 40 standard deviation']
sample_name_arr_df3 = ['Countries']


def data_filter(data, column_index_string, df):
    # Write your solution here and remove pass
    str1 = ""
    str2 = ""
    data1 = []
    for element in data.index:
        str1 = data.loc[element, column_index_string]
        if (str(str1).find('−') != -1):
            df3 = df.drop(element)
            # display(df3)
            df3 = df3.reset_index(drop=True)

            continue
        else:
            for i in range(len(str1)):
                if (str1[i] == "±"):
                    break
                str2 += str1[i]
            data1.append(float(str2))
            str2 = ""
    return (data1)


def data_filter2(data, column_index_string, df):
    # Write your solution here and remove pass
    str1 = ""
    str2 = ""
    data1 = []
    for element in data.index:
        str1 = data.loc[element, column_index_string]
        if (str(str1).find('−') != -1):
            df = df.drop(element)
            df = df.reset_index(drop=True)
            continue
        else:
            for i in range(len(str1)):
                if (str1[i] == "±"):
                    break
                str2 += str1[i]
            data1.append(float(str2))
            str2 = ""
    return (data1, df)


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
        if (str(str1).find('−') != -1):
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

    return (new_df)


def create_dataframe2(sample_name, column_name_arr_df2, row_type_arr, y_H_ex, y_H_ex_error, y_H_in, y_H_in_error):
    data = {sample_name: row_type_arr, column_name_arr_df2[0]: y_H_ex, column_name_arr_df2[1]: y_H_ex_error, column_name_arr_df2[2]: y_H_in, column_name_arr_df2[3]: y_H_in_error}
    new_df = pd.DataFrame(data)
    new_df = new_df.reset_index(drop=True)
    return ()


def create_dataframe3(sample_name, column_name_arr_df3, row_type_arr, Ra_226_val_arr, Ra_226_std_dev_arr, Th_232_val_arr, Th_232_std_dev_arr, K_40_val_arr, K_40_std_dev_arr):
    # np_arr = np.asarray(val_arr)
    # df = pd.DataFrame(np_arr,
    # columns=[column_name_val_arr])
    # initialize data of lists.
    data = {sample_name: row_type_arr, column_name_arr_df3[0]: Ra_226_val_arr, column_name_arr_df3[1]: Ra_226_std_dev_arr, column_name_arr_df3[2]
        : Th_232_val_arr, column_name_arr_df3[3]: Th_232_std_dev_arr, column_name_arr_df3[4]: K_40_val_arr, column_name_arr_df3[5]: K_40_std_dev_arr}
    new_df = pd.DataFrame(data)
    new_df = new_df.reset_index(drop=True)
    return (new_df)


def calc_Ra_eq(A_Ra, A_Th, A_K):
    Ra_eq = 370*((A_Ra/370) + (A_Th/259) + (A_K/4810))
    return (Ra_eq)


def get_Ra_eq_arr(df, column_1, column_2, column_3, index_lower, index_upper):
    Ra_eq_arr = []
    for element in range(index_lower, index_upper-1):
        Ra_eq_arr.append(calc_Ra_eq(
            df.loc[element, column_1], df.loc[element, column_2], df.loc[element, column_3]))
    return (Ra_eq_arr)


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

y_cement_Ra_226 = data_filter(df1[2:9], column_Ra_226_df1, df1)
y_cement_Ra_226.append(round((sum(y_cement_Ra_226)/len(y_cement_Ra_226)), 1))
y_cement_Th_232 = data_filter(df1[2:9], column_Th_232, df1)
y_cement_Th_232.append(round((sum(y_cement_Th_232)/len(y_cement_Th_232)), 1))
y_cement_K_40 = data_filter(df1[2:9], column_K_40, df1)
y_cement_K_40.append(round((sum(y_cement_K_40)/len(y_cement_K_40)), 1))

cement_df = create_dataframe1(sample_name_arr_df1[0], column_name_arr_df1, x_cement,  y_cement_Ra_226, error1_vektor_cement_Ra_226,
                              y_cement_Th_232, error2_vektor_cement_Th_232, y_cement_K_40, error3_vektor_cement_K_40)

fig, ax = plt.subplots(3)
cement_Ra_226_bar = ax[0].bar(x_cement, y_cement_Ra_226, yerr=error1_vektor_cement_Ra_226,
                              align='center', alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylabel("Ra 226 in Bq/kg")
ax[0].set_xlabel("cement type")
ax[0].set_title('Radioactive concetration of Ra 226 in different cement types')
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)

cement_Th_232_bar = ax[1].bar(x_cement, y_cement_Th_232, yerr=error2_vektor_cement_Th_232,
                              align='center', alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylabel("Th 232 in Bq/kg")
ax[1].set_xlabel("cement type")
ax[1].set_title('Radioactive concetration of Th 232 in different cement types')
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

cement_K_40_bar = ax[2].bar(x_cement, y_cement_K_40, yerr=error3_vektor_cement_K_40,
                            align='center', alpha=0.5, ecolor='black', capsize=10)
ax[2].set_ylabel("K 40 in Bq/kg")
ax[2].set_xlabel("cement type")
ax[2].set_title('Radioactive concetration of K 40 in different cement types')
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)

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

y_fly_ash_Ra_226 = data_filter(df1[11:16], column_Ra_226_df1, df1)
y_fly_ash_Ra_226.append(
    round((sum(y_fly_ash_Ra_226)/len(y_fly_ash_Ra_226)), 1))
y_fly_ash_Th_232 = data_filter(df1[11:16], column_Th_232, df1)
y_fly_ash_Th_232.append(
    round((sum(y_fly_ash_Th_232)/len(y_fly_ash_Th_232)), 1))
y_fly_ash_K_40 = data_filter(df1[11:16], column_K_40, df1)
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
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)

fly_ash_Th_232_bar = ax[1].bar(x_fly_ash, y_fly_ash_Th_232, yerr=error2_vektor_fly_ash_Th_232,
                               align='center', alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylabel("Th 232 in Bq/kg")
ax[1].set_xlabel("fly ash type")
ax[1].set_title(
    'Radioactive concetration of Th 232 in different fly ash types')
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

fly_ash_K_40_bar = ax[2].bar(x_fly_ash, y_fly_ash_K_40, yerr=error3_vektor_fly_ash_K_40,
                             align='center', alpha=0.5, ecolor='black', capsize=10)
ax[2].set_ylabel("K 40 in Bq/kg")
ax[2].set_xlabel("fly ash type")
ax[2].set_title('Radioactive concetration of K 40 in different fly ash types')
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)

ax[0].bar_label(fly_ash_Ra_226_bar, padding=3)
ax[1].bar_label(fly_ash_Th_232_bar, padding=3)
ax[2].bar_label(fly_ash_K_40_bar, padding=3)

fig1.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                     top=0.9, wspace=0.4, hspace=0.4)
fig1.set_size_inches(8, 16)
fig1.suptitle('fly ash')
fig1.savefig('figure1.png', dpi=300)

# create_dataframe(new_df_cement_sample_coloumn_name, x_cement, new_df_Ra_226_value_coloumn_name, y_cement_Ra_226, new_df_Ra_226_std_div_coloumn_name, error1_vektor_cement_Ra_226)
pie_fig, ax = plt.subplots(1)
ax = plt.pie(cement_df.loc[7, ["Ra 226 Value", "Th 232 Value", "K 40 Value"]], labels=[
             'Ra_226', 'Th_232', 'K_40'])
pie_fig.savefig('figure2.png', dpi=300)
# df2
# cement
y_cement_H_ex = data_filter(df2[2:10], column_H_ex, df2)
y_cement_H_ex_error = error_filter(df2[2:10], column_H_ex)
y_cement_H_in = data_filter(df2[2:10], column_H_in, df2)
y_cement_H_in_error = error_filter(df2[2:10], column_H_in)

df_cement_h = create_dataframe2(sample_name_arr_df2[0], column_name_arr_df2, x_cement,
                                y_cement_H_ex, y_cement_H_ex_error, y_cement_H_in, y_cement_H_in_error)

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
                      color='orange', align='center', alpha=0.5, capsize=5, bottom=b_threshold_Hex)
cement_bar_3 = ax.bar(x_cement_label + width/2, b_threshold_Hin,
                      width,  color='green', align='center', alpha=0.5, label='Hin')
cement_bar_4 = ax.bar(x_cement_label + width/2, a_threshold_Hin, width, yerr=y_cement_H_in_error,
                      color='red', align='center', alpha=0.5, capsize=5, bottom=b_threshold_Hin)

plt.axhline(threshold, color='black', ls='dashed', label="Grenzwert")

ax.set_xticks(x_cement_label, x_cement)
ax.legend()
plt.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("cement types")
plt.ylabel("Hazard Index")

# ax.bar_label(fly_ash_bar_1, padding=3)
ax.bar_label(cement_bar_2, padding=3)
# ax.bar_label(fly_ash_bar_3, padding=3)
ax.bar_label(cement_bar_4, padding=3)
plt.tight_layout()

threshold_fig_cement.savefig("figure3")

# fly ash
y_fly_ash_H_ex = data_filter(df2[11:17], column_H_ex, df2)
y_fly_ash_H_ex_error = error_filter(df2[11:17], column_H_ex)
y_fly_ash_H_in = data_filter(df2[11:17], column_H_in, df2)
y_fly_ash_H_in_error = error_filter(df2[11:17], column_H_in)

df_fly_ash_h = create_dataframe2(sample_name_arr_df2[1], column_name_arr_df2, x_fly_ash,
                                 y_fly_ash_H_ex, y_fly_ash_H_ex_error, y_fly_ash_H_in, y_fly_ash_H_in_error)

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
                       color='orange', alpha=0.5,  ecolor='black', capsize=5, bottom=b_threshold_Hex)
fly_ash_bar_3 = ax.bar(x_fly_ash_label + width/2, b_threshold_Hin,
                       width, color='green', align='center', alpha=0.5,  label='Hin')
fly_ash_bar_4 = ax.bar(x_fly_ash_label + width/2, a_threshold_Hin, width, yerr=y_fly_ash_H_in_error,
                       color='red', alpha=0.5, ecolor='black', capsize=5, bottom=b_threshold_Hin)

plt.axhline(threshold, color='black', ls='dashed', label="Grenzwert")

ax.set_xticks(x_fly_ash_label, x_fly_ash)
plt.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Fly ash types")
plt.ylabel("Hazard Index")

# ax.bar_label(fly_ash_bar_1, padding=3)
ax.bar_label(fly_ash_bar_2, padding=3)
# ax.bar_label(fly_ash_bar_3, padding=3)
ax.bar_label(fly_ash_bar_4, padding=3)

threshold_fig_fly_ash.tight_layout()
threshold_fig_fly_ash.savefig("figure4")
# df3

index_upper = 19
index_lower = 2

y_value_cement_Ra_226_df3, df3 = data_filter2(
    df3[index_lower:index_upper], column_Ra_226_df3, df3)
y_error_cement_Ra_226_df3 = error_filter(
    df3[index_lower:index_upper-1], column_Ra_226_df3)
y_value_cement_Th_232_df3, df3 = data_filter2(
    df3[index_lower:index_upper-1], column_Th_232_df3, df3)
y_error_cement_Th_232_df3 = error_filter(
    df3[index_lower:index_upper-1], column_Th_232_df3)
y_value_cement_K_40_df3, df3 = data_filter2(
    df3[index_lower:index_upper-1], column_K_40_df3, df3)
y_error_cement_K_40_df3 = error_filter(
    df3[index_lower:index_upper-1], column_K_40_df3)

x_cement_df3 = df3[index_lower:(index_upper-1)][column_country]

print(x_cement_df3)

df3_cement = create_dataframe3(sample_name_arr_df3[0], column_name_arr_df3, x_cement_df3, y_value_cement_Ra_226_df3,
                               y_error_cement_Ra_226_df3, y_value_cement_Th_232_df3, y_error_cement_Th_232_df3, y_value_cement_K_40_df3, y_error_cement_K_40_df3)
Ra_eq_arr = get_Ra_eq_arr(
    df3_cement, column_name_arr_df3[0], column_name_arr_df3[2], column_name_arr_df3[4], index_lower - 2, (index_upper - 2))
df3_cement['Ra eq'] = Ra_eq_arr
# print(Ra_eq_arr )

display(df3_cement)

sort_country_plot, ax = plt.subplots(1)

df3_cement = df3_cement.sort_values("Ra eq")
ax.barh(df3_cement[0:16]["Countries"], df3_cement[0:16]
        ["Ra eq"], align='center', alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("Ra eq. in Bq/kg")
plt.ylabel("Countries")
plt.title("Cement")
sort_country_plot.set_size_inches(16, 6)
sort_country_plot.savefig("figure5", dpi=300)

plt.show()
