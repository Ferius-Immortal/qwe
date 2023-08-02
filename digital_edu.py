import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv("train.csv")
df.drop(
    ["career_end", "career_start", "occupation_name", 
    "id", "people_main", "bdate",
    "has_photo", "has_mobile", "followers_count",
    "last_seen", "city", "graduation",
    "relation", "langs", "life_main"], axis=1, inplace=True)

df["education_form"].fillna("Full-time", inplace=True)
def education_form_hierarchy(edu_form):
    if edu_form == "Full-time":
        return 0
    elif edu_form == "Part-time":
        return 1
    elif edu_form == "Distance Learning":
        return 2
df["education_form"] = df["education_form"].apply(education_form_hierarchy)

def education_status_hierarchy(edu_status):
    if edu_status == "Undergraduate appliciant":
        return 0
    elif edu_status  == "Student (Specialist)" or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)":
        return 1
    elif edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)":
        return 2
    elif edu_status == "Candidate of Sciences":
        return 3
    else:
        return 4
df["education_status"] = df["education_status"].apply(education_status_hierarchy)

df[list(pd.get_dummies(df['occupation_type']).columns)] = pd.get_dummies(df['occupation_type'])
df.drop(["occupation_type"], axis=1, inplace=True)


x = df.drop("result", axis = 1)
y = df["result"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

percent = accuracy_score(y_test, y_pred) * 100
print(percent)

df.info()