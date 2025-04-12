import streamlit as st
from streamlit_option_menu import option_menu
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from pandas import concat
import keras as keras
from joblib import load

encoder = load("encoder.pkl")
pca = load("pca_model.pkl")


def prepare_input_for_prediction(
    experience: float,
    age: str,
    education: str,
    job_title: str,
    gender: str
):
    
    scaler = MinMaxScaler()
    user_input = {
        "Years of Experience": experience,
        "Age": age,
        "Education Level": education,
        "Job Title": job_title,
        "Gender": gender
    }
    input_df = DataFrame([user_input])

    category_columns = ["Age", "Education Level", "Job Title","Gender"]
    numeric_columns = ["Years of Experience"]

    encoded_array = encoder.transform(input_df[category_columns])
    
    encoded_col_names = encoder.get_feature_names_out()

    print(encoded_col_names)

    encoded_df = DataFrame(encoded_array, columns=encoded_col_names)

    final_df = concat([input_df[numeric_columns].reset_index(drop=True), encoded_df], axis=1)

    final_scaled = scaler.fit_transform(final_df)

    final_pca = pca.transform(final_scaled)

    return final_pca

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Model Description", "Model Execution"],
        icons=["info-circle", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )


if selected == "Model Description":
    st.title("🧠 Model Description")
    st.markdown("""
    This app demonstrates a simple ML model.

    **Details:**
    - Model Type: ANN or Regression
    - Dataset: Salary, MNIST, etc.
    - Purpose: Predict salary / classify digit / other logic
    """)

elif selected == "Model Execution":
    gender_options = ["Male", "Female"]

age_options = [str(i) for i in range(23, 54)]

education_options = ["Bachelor's", "Master's", "PhD"]


job_titles = [
    'Account Manager', 'Accountant', 'Administrative Assistant', 'Business Analyst',
    'Business Development Manager', 'Business Intelligence Analyst', 'CEO',
    'Chief Data Officer', 'Chief Technology Officer', 'Content Marketing Manager',
    'Copywriter', 'Creative Director', 'Customer Service Manager', 'Customer Service Rep',
    'Customer Service Representative', 'Customer Success Manager', 'Customer Success Rep',
    'Data Analyst', 'Data Entry Clerk', 'Data Scientist', 'Digital Content Producer',
    'Digital Marketing Manager', 'Director', 'Director of Business Development',
    'Director of Engineering', 'Director of Finance', 'Director of HR',
    'Director of Human Capital', 'Director of Human Resources', 'Director of Marketing',
    'Director of Operations', 'Director of Product Management', 'Director of Sales',
    'Director of Sales and Marketing', 'Event Coordinator', 'Financial Advisor',
    'Financial Analyst', 'Financial Manager', 'Graphic Designer', 'HR Generalist',
    'HR Manager', 'Help Desk Analyst', 'Human Resources Director', 'IT Manager',
    'IT Support', 'IT Support Specialist', 'Junior Account Manager', 'Junior Accountant',
    'Junior Advertising Coordinator', 'Junior Business Analyst',
    'Junior Business Development Associate', 'Junior Business Operations Analyst',
    'Junior Copywriter', 'Junior Customer Support Specialist', 'Junior Data Analyst',
    'Junior Data Scientist', 'Junior Designer', 'Junior Developer',
    'Junior Financial Advisor', 'Junior Financial Analyst', 'Junior HR Coordinator',
    'Junior HR Generalist', 'Junior Marketing Analyst', 'Junior Marketing Coordinator',
    'Junior Marketing Manager', 'Junior Marketing Specialist', 'Junior Operations Analyst',
    'Junior Operations Coordinator', 'Junior Operations Manager', 'Junior Product Manager',
    'Junior Project Manager', 'Junior Recruiter', 'Junior Research Scientist',
    'Junior Sales Representative', 'Junior Social Media Manager',
    'Junior Social Media Specialist', 'Junior Software Developer',
    'Junior Software Engineer', 'Junior UX Designer', 'Junior Web Designer',
    'Junior Web Developer', 'Marketing Analyst', 'Marketing Coordinator',
    'Marketing Manager', 'Marketing Specialist', 'Network Engineer', 'Office Manager',
    'Operations Analyst', 'Operations Director', 'Operations Manager',
    'Principal Engineer', 'Principal Scientist', 'Product Designer', 'Product Manager',
    'Product Marketing Manager', 'Project Engineer', 'Project Manager',
    'Public Relations Manager', 'Recruiter', 'Research Director', 'Research Scientist',
    'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager',
    'Sales Operations Manager', 'Sales Representative', 'Senior Account Executive',
    'Senior Account Manager', 'Senior Accountant', 'Senior Business Analyst',
    'Senior Business Development Manager', 'Senior Consultant', 'Senior Data Analyst',
    'Senior Data Engineer', 'Senior Data Scientist', 'Senior Engineer',
    'Senior Financial Advisor', 'Senior Financial Analyst', 'Senior Financial Manager',
    'Senior Graphic Designer', 'Senior HR Generalist', 'Senior HR Manager',
    'Senior HR Specialist', 'Senior Human Resources Coordinator',
    'Senior Human Resources Manager', 'Senior Human Resources Specialist',
    'Senior IT Consultant', 'Senior IT Project Manager', 'Senior IT Support Specialist',
    'Senior Manager', 'Senior Marketing Analyst', 'Senior Marketing Coordinator',
    'Senior Marketing Director', 'Senior Marketing Manager',
    'Senior Marketing Specialist', 'Senior Operations Analyst',
    'Senior Operations Coordinator', 'Senior Operations Manager',
    'Senior Product Designer', 'Senior Product Development Manager',
    'Senior Product Manager', 'Senior Product Marketing Manager',
    'Senior Project Coordinator', 'Senior Project Manager',
    'Senior Quality Assurance Analyst', 'Senior Research Scientist', 'Senior Researcher',
    'Senior Sales Manager', 'Senior Sales Representative', 'Senior Scientist',
    'Senior Software Architect', 'Senior Software Developer', 'Senior Software Engineer',
    'Senior Training Specialist', 'Senior UX Designer', 'Social Media Manager',
    'Social Media Specialist', 'Software Developer', 'Software Engineer',
    'Software Manager', 'Software Project Manager', 'Strategy Consultant',
    'Supply Chain Analyst', 'Supply Chain Manager', 'Technical Recruiter',
    'Technical Support Specialist', 'Technical Writer', 'Training Specialist',
    'UX Designer', 'UX Researcher', 'VP of Finance', 'VP of Operations',
    'Web Developer'
]

with st.form("input_form"):
    st.title("📝 Candidate Info Form")

    gender = st.selectbox("Gender", gender_options)
    age = st.selectbox("Age", age_options)
    education = st.selectbox("Education Level", education_options)
    job_title = st.selectbox("Job Title", job_titles)
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

    submit_btn = st.form_submit_button("Submit")

if submit_btn:
    st.success("✅ Form submitted successfully!")

    model = keras.models.load_model("salary_model.keras")

    data = prepare_input_for_prediction(experience,age,education,job_title,gender)

    scaler = load("arget_salary_scaler.pkl")
    prediction = model.predict(data)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

    st.markdown(f"""
    **Your Input Summary:**
    - 👤 Gender: `{gender}`
    - 🎂 Age: `{age}`
    - 🎓 Education: `{education}`
    - 💼 Job Title: `{job_title}`
    - 🧮 Experience: `{experience} years`
    - 🔍 Model Prediction: `{prediction}`
    """)


