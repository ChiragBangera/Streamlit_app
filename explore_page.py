import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_ed(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than Bachelor’s'


@st.cache
def load_data():
    raw_data = pd.read_csv('survey_results_public.csv')
    data = raw_data[['Employment', 'Country',
                     'EdLevel', 'YearsCodePro', 'ConvertedComp']]
    data = data.rename({'ConvertedComp': 'Salary'}, axis=1)
    data = data.dropna()
    data_1 = data[data['Employment'] == 'Employed full-time']
    data_1 = data_1.drop(['Employment'], axis=1)
    data_2 = data_1[data_1['Country'].map(
        data_1['Country'].value_counts()) > 400]
    data_3 = data_2[data_2['Salary'] < 250000]
    data_3 = data_3[data_3['Salary'] > 10000]
    data_3['YearsCodePro'] = data_3['YearsCodePro'].apply(clean_experience)
    data_3['EdLevel'] = data_3['EdLevel'].apply(clean_ed)
    return data_3


dataframe = load_data()


def show_explore_page():
    st.title('Explore software Engineer Salaries')
    st.write(
        """### Stack overflow Developer Survey 2020"""
    )

    data = dataframe['Country'].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%",
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn a a circle.
    ax1.axis("equal")

    st.write("""### Number of Data from different countries""")
    st.pyplot(fig1)

    st.write("""#### Mean Salary Based on Country""")
    dat1 = dataframe.groupby(['Country'])[
        'Salary'].mean().sort_values(ascending=True)

    st.bar_chart(dat1)

    st.write("""#### Mean Salary Based on Experience""")
    dat2 = dataframe.groupby(['YearsCodePro'])[
        'Salary'].mean().sort_values(ascending=True)
    st.line_chart(dat2)
