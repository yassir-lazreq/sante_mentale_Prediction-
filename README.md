# 🧠 Student Mental Health Risk Prediction

> A structured, multi-feature classification dataset for predicting mental health risk levels among university students.

---

## 📋 Problem Statement

### Context
University students face a confluence of stressors — academic pressure, financial strain, social isolation, irregular sleep, and lifestyle disruptions — that place them at disproportionately high risk for mental health challenges. Despite growing awareness, early identification of at-risk students remains difficult due to the multifactorial nature of mental wellbeing.

### Objective
Build a machine learning model that predicts the **mental health risk level** (Low / Medium / High) of a university student based on behavioral, academic, lifestyle, and psychosocial features collected through a simulated student survey.

### Why It Matters
Mental health conditions affect approximately **1 in 3 university students** globally, yet most go undetected until they reach a crisis point. A predictive model could enable universities to proactively identify vulnerable students and offer early intervention — improving academic outcomes and student wellbeing simultaneously.

---

## 📁 Dataset Overview

| Property | Value |
|---|---|
| Rows | 2,000 |
| Columns | 17 (16 features + 1 target) |
| Task Type | Multi-class Classification |
| Target Variable | `Mental_Health_Risk` (Low / Medium / High) |
| Missing Values | Yes (~3–5% in selected columns) |

---

## 🗂️ Column Documentation

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `Student_ID` | string | Unique student identifier | STU0001 |
| `Age` | integer | Student age in years | 18–30 |
| `Gender` | categorical | Student gender identity | Male, Female, Non-binary |
| `Year_of_Study` | integer | Current academic year | 1–5 |
| `Sleep_Hours` | float | Average nightly sleep (hours) | 3.5–9.5 |
| `Study_Hours_Per_Week` | float | Self-reported study hours per week | 5–55 |
| `GPA` | float | Current cumulative GPA (4.0 scale) | 1.5–4.0 |
| `Social_Media_Hours` | float | Average daily social media usage (hours) | 0.5–10.0 |
| `Screen_Time_Hours` | float | Total daily screen time (hours) | 1.0–14.0 |
| `Physical_Activity` | categorical | Weekly exercise frequency level | Low, Moderate, High |
| `Financial_Stress` | float | Self-rated financial stress (1–10 scale) | 1.0–10.0 |
| `Family_Support` | float | Perceived family support level (1–10) | 1.0–10.0 |
| `Attendance_Rate` | float | Class attendance percentage | 30–100 |
| `Part_Time_Job` | binary | Whether student holds a part-time job | Yes, No |
| `Diet_Quality` | categorical | Self-reported diet quality | Poor, Average, Good |
| `Previous_Mental_Health_History` | binary | Prior diagnosed mental health condition | Yes, No |
| `Mental_Health_Risk` | categorical | **TARGET** — predicted risk level | Low, Medium, High |

