---
layout: post
title: Using Machine Learning to Predict Winners in NCAA's March Madness - Part 1 - Overview
published: true
---
##  Using Machine Learning to Predict Winners in NCAA's March Madness - Part 1: Overview

### Overview:

March Madness rolls around every year, and every year I sit back and watch my friends get involved in bracket madness while I remember (with regret) the year I filled out a bracket. That year, I did not know that the numbers to the side of each team's name were called 'seeds,' and that those seeds were predictive of the team's likelihood of surviving through six rounds of matches. I chose my bracket winners based on chance, and therefore my bracket was about as good as a horse predicting the winners of a three-legged race by prancing on a Twister board.

This analysis will involve a little more attention to seeds. In fact, I will start off by looking at how accurate the seeds are in predicting the winners over the previous 33 seasons. Let's start off by introducing the basics of the March Madness teams and tournament.


### The Teams:
There are 32 division conference winners that go to the tournament, along wth 36 at-large conference champions. Four of those 68 seeds actually come from matches between low-seeded teams. The NCAA had this to say about the play-in games:

*When the Division I men’s basketball committee decided to expand the field to 68 teams in advance of the 2011 tournament, they came up with the First Four concept and initially looked at two formats.
The first involved having the last eight teams on the overall seed list (which is a ranking of all 68 teams) to play four games to determine the four No. 16 seeds. The other concept was to take the last eight at-large selections and have those teams play four games to advance.
Eventually they decided on a hybrid of the two options, with the last four at-large teams and the last four on the overall seed list playing for the right to advance to the second round.*

### The tournament:

The tournament begins with the "First Four" play-in games that will determine the final seeds for lower ranking, at-large teams. Once those matches are comlete, the official "First Round" begins and teams compete in 8 matches in each of four NCAA-designated regions. Then follows the "Second Round," the "Sweet Sixteen," the "Elite Eight," the "Final Four" and the championship big finale.

I'm not completely sure what I'll find when I look at the data, so I'll import some of the tools we need for our Exploratory Data Analysis. EDA is important at this step. We use EDA to find out what relationships exist between our target (outcome variable, label, y) and our features (predictor variables, X), as well as discover if there are any important relationships between our features that require some finessing (i.e. multicollinearity that will wreck our model).  

### Imports:

I'm going to take a stab in the dark and import some tools for the actual modeling and model evaluation. 

      # 	Data importing, cleaning & EDA
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      import seaborn as sns
      import holoviews as hv
      
      %matplotlib inline

      # 	Data train test splitting
      from sklearn.model_selection import train_test_split
      
      # 	Preprocessing  pipelines
      from sklearn.pipeline import Pipeline, make_pipeline
      
      # 	PCA Modeling
      from sklearn.svm import SVC
      from sklearn.decomposition import PCA
      
      # 	Logisti Regression Modeling
      from sklearn.linear_model import LogisticRegression
      
      # 	Model Tuning, Regularization & Evaluation
      from sklearn.model_selection import GridSearchCV

### Next Steps: Getting our Data:

The data import will be the next blog in this series...
