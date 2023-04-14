import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder


class DataLoader:
    def load_data(self):
        # Load dataset
        df_group8 = pd.read_csv("KSI.csv")

        # Describe data
        print("Data shape:", df_group8.shape)
        print("Data columns:", df_group8.columns)
        print()
        print("Data summary:")
        print(df_group8.describe())

        # datatype and ranges
        df_group8.info()
        print()

        # Unique values in categorical columns
        print("Unique values in categorical columns:")
        for col in df_group8.columns:
            if df_group8[col].dtype == "object":
                unique_vals = df_group8[col].unique()
                print(f"{col}: {unique_vals}")

        # check for missing values
        print(df_group8.isnull().sum())

        # Replace Property Damange Only with Non-Fatal Injury
        df_group8.replace("Property Damage Only",
                          "Non-Fatal Injury", inplace=True)

        # Create month column extracted from Date
        df_group8["DATE"] = pd.to_datetime(df_group8["DATE"])
        df_group8["MONTH"] = df_group8["DATE"].dt.month
        df_group8["DAY_WEEK"] = df_group8["DATE"].dt.dayofweek

        # Add "No" value to the Yes/No fields
        columns_yes_no = [
            "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK", "TRSN_CITY_VEH", "EMERG_VEH", "PASSENGER", "SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY"
        ]
        df_group8[columns_yes_no] = df_group8[columns_yes_no].fillna("No")

        # Fields to remove
        columns_to_remove = [
            "X",  # Same with latitude / longitude
            "Y",  # Same with latitude / longitude
            "INDEX_",  # A unique identifier of the inputs
            "ACCNUM",  # A unique identifier for each accident
            "STREET1",  # Too many unique value
            "STREET2",  # Too many unique value
            "OFFSET",  # Too many null value and unique category
            "DATE",  # Duplicate field - Year exists and road surface condition to identify the season
            "TIME",  # Unncessary field due to light, visibility, and road condition
            "ACCLOC",  # Too many null value and duplicate to LOCCOORD
            "PEDTYPE",  # Too many null value
            "PEDACT",  # Too many null value
            "PEDCOND",  # Too many null value
            "CYCLISTYPE",  # Too many null value
            "CYCACT",  # Too many null value
            "CYCCOND",  # Too many null value
            "HOOD_158",  # Too many unique value
            "NEIGHBOURHOOD_158",  # Too many unique value
            "HOOD_140",  # Too many unique value
            "NEIGHBOURHOOD_140",  # Too many unique value
            "ObjectId",  # Too many unique value

            "INJURY",  # Rather result than feature
            "FATAL_NO",  # Rather result than feature. Unique
        ]
        df_group8.drop(columns=columns_to_remove, inplace=True)

        # Remove rows with missing values in the target column
        df_group8.dropna(subset=["ACCLASS"], inplace=True)

        # Encode
        df_group8["ACCLASS"].replace({"Fatal": 1, "Non-Fatal Injury": 0}, inplace=True)

        # Separate the feature and target columns
        X = df_group8.drop("ACCLASS", axis=1)
        y = df_group8["ACCLASS"]

        # Create a pipeline to streamline feature transformation and selection
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Use Ordinal Encoder for Feature Selection
            ("ordinal", OrdinalEncoder())
        ])

        numerical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])

        preprocessor = ColumnTransformer([
            ("cat", categorical_transformer, selector(dtype_include="object")),
            ("num", numerical_transformer, selector(dtype_exclude="object"))
        ], remainder="drop")

        X_transformed = preprocessor.fit_transform(X)

        # Split the data into 10% test and 90% training
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.1, stratify=y, random_state=13)

        # Use Random Forest's Feature Importance to select the important features
        forest = RandomForestClassifier(
            random_state=13, n_jobs=-2, class_weight="balanced")
        forest.fit(X_train, y_train)

        importances = forest.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in forest.estimators_], axis=0)

        feat_list = []

        total_importance = 0
        # Print the name and gini importance of each feature
        for feature in zip(X.columns, importances):
            feat_list.append(feature)
            total_importance += feature[1]

        most_important_features = []
        # Print the name and gini importance of each feature
        for feature in zip(X.columns, importances):
            if feature[1] > .017:
                most_important_features.append(feature[0])

        # create DataFrame using data
        df_imp = pd.DataFrame(feat_list, columns=["FEATURE", "IMPORTANCE"]).sort_values(
            by="IMPORTANCE", ascending=False)
        df_imp["CUMSUM"] = df_imp["IMPORTANCE"].cumsum()
        print(df_imp)

        # Most important feature selected are the features with more than .017 importance
        print("Most Important Features:", most_important_features)
        print("Number of Included Features =", len(most_important_features))

        # Subset data with important features
        df_group8_feature_selected = df_group8[most_important_features]

        # Re-organize then address imbalances
        categorical_transformer_ohe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("cat", categorical_transformer_ohe, selector(dtype_include="object")),
            ("num", numerical_transformer, selector(dtype_exclude="object"))
        ], remainder="drop")

        # fit and transform the important features
        df_group8_feature_selected_transformed = preprocessor.fit_transform(
            df_group8_feature_selected)

        # Split the data into 10% test and 90% training
        X_train, X_test, y_train, y_test = train_test_split(
            df_group8_feature_selected_transformed, y, test_size=0.1, stratify=y, random_state=13)

        # Address the imbalance of classes
        oversampler = SMOTE()
        X_train_oversampled, y_train_oversampled = oversampler.fit_resample(
            X_train, y_train)

        # Save the data
        self.preprocessor = preprocessor
        self.X_train_oversampled = X_train_oversampled
        self.y_train_oversampled = y_train_oversampled
        self.X_test = X_test
        self.y_test = y_test
        self.df = df_group8_feature_selected

    def get_unique_values_by_features(self) -> dict[str, list[str]]:
        """
        Return a dictionary of features and their unique values
        """
        df = self.df
        feature_by_uniques = {}
        for col in df.columns:
            values: list = df[col].unique().tolist()

            # If continuous numbers
            if 80 <= len(values):
                # Only the min and max values
                feature_by_uniques[col] = [min(values), max(values)]
                continue

            # If all are numbers, sort as numbers
            if all(isinstance(x, (int, float)) for x in values):
                values.sort()
            # Else sort as strings
            else:
                values.sort(key=lambda x: str(x))

            feature_by_uniques[col] = values

        return feature_by_uniques
