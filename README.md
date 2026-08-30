# UI & Learning Study: Analysis Pipeline

Data cleaning and statistical analysis for an independent research project examining how digital interface design affects student engagement, accuracy, and persistence.

## Overview

This study tested whether three different interface designs — **minimal**, **scaffolded**, and **gamified** — produce different levels of engagement, accuracy, and time-on-task among students aged 13–17 completing the same logic puzzle. 90 participants (n = 30 per condition) were recruited and randomly assigned to one of the three versions.

**Hypotheses:**
- H1: Gamified and scaffolded layouts will produce higher engagement and persistence than the minimal layout.
- H2: Prior experience with digital learning platforms will positively correlate with engagement.

## What this script does

1. Loads and merges three condition-specific response datasets (exported from Google Forms)
2. Standardizes inconsistent column naming and response formatting across the three forms (each form independently used slightly different question labels and response text)
3. Converts Likert-scale responses into numeric scores and computes a composite engagement measure
4. Converts categorical time-spent responses into numeric estimates
5. Parses raw puzzle-accuracy scores
6. Runs one-way ANOVA and Tukey HSD post-hoc tests across the three conditions
7. Runs a linear regression testing prior experience as a predictor of engagement
8. Generates and exports all result visualizations (bar charts, boxplot, regression scatter)

## Findings (summary)

No statistically significant differences were found between conditions on engagement (p = .062), accuracy (p = .194), or time-on-task (p = .781), and prior experience did not significantly predict engagement (p = .892). However, substantial within-condition variability in engagement scores was observed, suggesting individual differences among learners may play a larger role in engagement than interface design alone — a finding that motivated exploring adaptive, personalized design (see [Compass Code](#), a related project).

## Files

- `ui_learning_study_analysis.py` — full analysis pipeline (Python, run in Google Colab)

## Tools used

Python · pandas · scipy · statsmodels · seaborn · matplotlib · Google Colab

## Live notebook

https://colab.research.google.com/drive/1Md9ktZp0w0QnUsBDLQbnU7D5RSt7qSAV?usp=sharing

---
*Independent research project by Ellie Shea. AI assistance was used in developing the data-cleaning and statistical analysis pipeline; all data collection, interpretation, and writing were conducted independently.*
