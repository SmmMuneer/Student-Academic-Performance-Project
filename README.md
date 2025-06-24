# Student-Academic-Performance-Project
Predicting Student Academic Performance Based on Sleep and Study Hours using Data Analysis and Visualization.

This project explores the impact of **weekly self-study hours** and **daily sleep hours** on students' academic performance using **Python**, **Pandas**, **Matplotlib**, and **Seaborn**. The dataset includes 2000 student records, with academic scores in Physics, Chemistry, and Biology, and computes the average performance to analyze patterns and relationships.


## 📊 Dataset Features

| Column Name                  | Description                              |
|-----------------------------|------------------------------------------|
| gender                      | 0 = Female, 1 = Male                     |
| extracurricular_activities  | 0 = No, 1 = Yes                          |
| weekly_self_study_hours     | Hours spent in self-study per week       |
| Daily sleep hours           | Average daily sleep duration in hours    |
| physics_score               | Physics subject score (0-100)            |
| chemistry_score             | Chemistry subject score (0-100)          |
| biology_score               | Biology subject score (0-100)            |
| Average(Academic Performance)| Average of three subject scores          |
| Achievement                 | High / Low based on performance          |
| achievement_binary          | 1 = High, 0 = Low                        |

---

## 📈 Key Analyses Performed

- **Descriptive statistics**: Mean, median, mode, standard deviation
- **Visualizations**:
  - Distribution plots of key features
  - Scatter plots of study/sleep vs. performance
  - Standard deviation thresholds
- **Hypothesis testing** using **t-test** for sleep hours effect
- **Model Training** (optional): Simple regression models using study and sleep hours

---

## 🧠 Project Objective

To analyze how much **study time** and **sleep duration** impact **academic success**, and visualize these relationships clearly to aid educational decision-making.


## 🛠️ Tech Stack

- Python 
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Jupyter Notebook


## 📂 How to Run

1. Clone the repository:
   run python app.py
