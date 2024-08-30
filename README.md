# Predicting the Primary Contributory Cause of Car Accidents

<img src="./Chicago_image.jpg" style="width:100%; height:auto;">

This project aims to develop a predictive model to identify the primary contributory causes of car accidents in the City of Chicago. The project utilizes advanced machine learning techniques, including classification algorithms and feature engineering, to create a robust model that can provide insights into accident causation.

## Objectives

* Systematically explore and clean the Chicago crash datasets to address missing values, outliers, and inconsistencies, ensuring the data is accurate and reliable for modeling. This process will establish a strong foundation for feature selection and model development.

* Identify and engineer the most relevant features that significantly influence crash causes, and reduce dimensionality where necessary to enhance model performance and interpretability. This step ensures the model focuses on the most critical factors, improving predictive accuracy.

* Develop and train a robust multi-class classification model to accurately predict the primary contributory cause of car accidents, experimenting with various algorithms like Random Forest and Logistic Regression to identify the best-performing approach.

* Rigorously evaluate the model’s performance using key metrics such as F1-score, precision, and recall, and optimize it through hyperparameter tuning and cross-validation to ensure it generalizes effectively to new, unseen data.

* Analyze the model’s output to uncover patterns or common factors linked to specific crash causes, providing actionable insights that can guide policy decisions, enhance road safety measures, and shape public awareness campaigns.

* Plan for the model’s integration into a decision-support system, making it accessible to city planners, law enforcement, and public safety officials to aid in strategic planning and proactive accident prevention efforts.

## Data Choice and Understanding

* Due to the large size of the available Crash Data files and limited resources, such as time for running and tuning models, we have decided to use only two datasets from the collection for our modeling efforts. This approach allows us to focus on achieving accurate results within our resource constraints while still leveraging a representative portion of the data. It also allows us to focus on the most relevant data while managing computational efficiency. The column `INJURY_CLASSIFICATION` from the `People` dataset will serve as our `Target variable` in the modelling process.

1. `Traffic_Crashes_-_People_20240824.csv` from  [ Driver/Passenger Data](https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d/about_data): This data contains information about `people` involved in a crash and if `any injuries` were sustained.

2. `Traffic_Crashes_-_Vehicles_20240824.csv` from [ Vehicle Data](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data): This dataset contains information about `vehicles` (or units as they are identified in crash reports) involved in a traffic crash. 

## EDA

#### The gender distribution

* Let's check the distribution of gender in the involvement of crashes or accidents.

<img src="Images/Distribution of Gender.png">

* The datasets had a larger sample of males involved in accidents compared to other gender with a leading percentage of `52.4%` which is more than half of the dataset.
* The females had a `38.2%` while X a non-binary or gender non-conforming identity had `9.4%`

#### Classification of Injuries by Gender

![Count of Injury Classifications by Sex](Images/Count of Injury Classifications by Sex.png)

* This graph shows the imbalance in the classifications with the class `NO_INDICATION_OF_INJURY` getting a large sample and `FATAL` getting the smallest sample.
* From the graph `MALES` have high occurrences in all the five classes followed by `FEMALES`.

#### Age_groups With the Most Accident Occurrences.

<img src="Images/Count_of_Males_and_Females_in_Each_Age_Group.png"><br>

* From the above graph, the individuals both `males` and `females` between age `19-30` were the most involved in car crashes.
* `Males` in this age group had the highest occurrences with over `180000` crashes while over `170000 Females` in the same age_group of `19-30` had a crash.

#### Injury Clasifications

![Percentage of Injury Classifications](Images/Percentage_of_Injury_Classifications.png)

* From the above graph we can see that the column `INJURY_CLASSIFICATION` has most of the records in the category `NO_INDICATION_OF_INJURY`

#### Classification of Physical Condition

* The `NORMAL` classification had the highest percentage with over 65% which shows that more than 65% crashes people involved were in normal conditions.

![Percentage of Physical Condition Classifications](Images/Percentage_of_Physical_Condition_Classifications.png)

## Models

* We started with a dummy