import pandas as pd

# df = pd.read_csv("data/creditcard.csv")
# print(df.head())
# print(df.info())
# print(df.describe())

# print("Number of samples : ",df.shape[0])
# print("Number of columns : ",df.shape[1])
# print("Number of Features : ", df.columns.tolist())
# print("Class Distribution : ",df["Class"].value_counts())
# print("Missing Values : ",df.isnull().sum())
# print("Number of duplicats ",df.duplicated().sum())

# print(df[df.duplicated()].head())

# print(df[df.duplicated(keep=False)].sort_values(
#     by=list(df.columns)
# ).head(20))

# print("_"*100)

#Total duplicate
# duplicate = df[df.duplicated(keep=False)]
# print("Total duplicate row: ",len(duplicate))
# print(duplicate["Class"].value_counts())

# print("_"*100)

# #Delet dupilcate
# print("Befor removing duplicate : ",df.shape)
# df = df.drop_duplicates()
# print("After removing duplicate : ",df.shape)
# print(df["Class"].value_counts())


#_______________________________________________________________

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data():
    
    df = pd.read_csv("data/creditcard.csv")
    df = df.drop_duplicates()

    
    X = df.drop("Class",axis=1)
    y = df["Class"]


    X_train , X_test , y_train , y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    X_train_raw , X_test_raw , y_train , y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    
    

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train , X_test , y_train , y_test , scaler , X_train_raw , X_test_raw