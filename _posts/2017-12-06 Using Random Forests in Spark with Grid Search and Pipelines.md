

## Overview
For this practice project, we are going to predict the acceleration of cars using data provided by Gareth James and Co at USC, found in the dataset `Auto.csv`, which can be downloaded at <a href="http://www-bcf.usc.edu/~gareth/ISL/data.html" target="_blank" rel="noopener">http://www-bcf.usc.edu/~gareth/ISL/data.html</a>.

A great place to get practice using Apache Spark and writing Scala scripts is on <a href="https://databricks.com/" target="_blank" rel="noopener">DataBricks</a>. I use a Scala notebook in this practice example.

You can sign up for the community edition, which is free.

## Step 1: To sign up, visit the DataBricks site and sign up for an account:
<img class="alignnone size-full wp-image-47" src="https://robotallie.files.wordpress.com/2017/12/databrickshome1.jpeg" alt="DatabricksHome1" width="1223" height="520" />

 

## Step 2: Start Today - Register with your contact information.

 

<img class="alignnone size-full wp-image-49" src="https://robotallie.files.wordpress.com/2017/12/databricks-step2.jpeg" alt="Databricks-Step2" width="1184" height="550" />

 

## Step 3: After your have confirmed your account, on the Home dashboard of DataBricks you will select "DATA" to upload the Auto.csv dataset you downloaded from USC.

 

<img class="alignnone size-full wp-image-50" src="https://robotallie.files.wordpress.com/2017/12/databricksstep3.jpeg" alt="DataBricksStep3" width="468" height="544" />

 

## Step 4: Either drag & drop the file (if you use Chrome) into the box to automatically upload it, or click on the box and search for the file in your file system.

 

<img class="alignnone size-full wp-image-51" src="https://robotallie.files.wordpress.com/2017/12/databricksstep5.jpeg" alt="DataBricksStep5" width="583" height="670" />

 

## Step 5: Once the file has been uploaded to DataBricks, you will see a green checkmark above the file and the filepath to access the file when you want to load the data into your notebook. You must copy this filepath and save it for later.

 

<img class="alignnone size-full wp-image-52" src="https://robotallie.files.wordpress.com/2017/12/databricksstep5b.jpeg" alt="DataBricksStep5B" width="522" height="673" />

## Step 6: You are ready to create a Scala notebook. Click on the "WORKSPACE" button on the left side-bar to take you to the dashboard of users and notebooks.

 

<img class="alignnone size-full wp-image-53" src="https://robotallie.files.wordpress.com/2017/12/databricksstep6.jpeg" alt="DataBricksStep6" width="522" height="673" />

## Step 7: On the WORKSPACE dashboard:
<ol>
	<li>Click on "Users" on the Workspace tab.</li>
	<li>Then click on your "username" on the Users tab.</li>
	<li>Click on the "down arrow" on your username to get the drop-down list of options.</li>
	<li>Click on "Create" until the next drop-down menu opens.</li>
	<li>Choose "Scala"</li>
</ol>
<img class="alignnone size-full wp-image-54" src="https://robotallie.files.wordpress.com/2017/12/databricksstep7.jpeg" alt="DataBricksStep7" width="1078" height="562" />

Once you have completed this, you are ready to set up your data and Apache Spark package imports.

In this next section, we are going to import the data and the Apache Spark packages that are necessary for:
<ul>
	<li>Data cleaning</li>
	<li>Feature engineering</li>
	<li>Model building</li>
	<li>Model evaluation</li>
</ul>
You may notice that I didn't mention anything about EDA here. This is because we expect that you will first do some data munging and exploration in R or Python before coming to Spark. Spark is best used as a place to put your finalized models into production. That means we will have already have decided which features to include in our analysis and we come to the table knowing which features need to be cleaned up a bit before they can fit into our pipeline.

## 1. Spark Package Imports</h3>
Let's start with importing Apache Spark packages for SQL (.sql) and for Machine Learning (.ml).

* First, for cleaning the data, we need to import packages related to regex because we're going to extract the make of the cars from the "name" column in the dataset that has the full name of cars. So import <code>regexp_extract, regexp_replace</code>.

* Next, for feature engineering, we will use the <code>OneHotEncoder, StringIndexer, VectorAssembler</code> packages

* Then, to make instantiate a random forest model, we will import the <code>RandomForestRegressor, Pipeline</code> packages.

* Lastly for the model evaluation, we will try out the cross validation package and use the evaluator for regression, so we can import `CrossValidator`, `ParamGridBuilder`, `RegressionEvaluator` to get those.

`// Cleaning the data
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions.{regexp_extract, regexp_replace}

// Feature engineering
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler}

// Model Building
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.Pipeline

// Model Evaluation
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}`

## 2. File I/O: 
- In the first few steps of setting up our data in DataBricks, we saved the filepath of the `Auto.csv` dataset after we uploaded it into our DataBricks dashboard. Have that ready now.
- Import the file into your Scala notebook using the filepath inside quotes for the .load() method. One of the things that is different about working in Scala if you are used to Python is indentation, and what that indentation means. In Scala, you can put chained methods on subsequent lines of code, rather than in one long line.

So this hard to read data import:

`var df = sqlContext.read.format("csv").option("header", "true").option("inferSchema","true").load("/FileStore/tables/Auto.csv")df.show()`

becomes this easy-to-read snippe:

`// /FileStore/tables/Auto.csv
var df = sqlContext
  .read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/Auto.csv")`

Make sure the data loaded correctly by checking the data types of each column and looking at the top 20 rows of the data.

`df.dtypes

res: df:org.apache.spark.sql.DataFrame
mpg:double
cylinders:integer
displacement:double
horsepower:integer
weight:integer
acceleration:double
year:integer
origin:integer
name:string
make:string`

`df.show()
+----+---------+------------+----------+------+------------+----+------+--------------------+
| mpg|cylinders|displacement|horsepower|weight|acceleration|year|origin|                name|
+----+---------+------------+----------+------+------------+----+------+--------------------+
|18.0|        8|       307.0|       130|  3504|        12.0|  70|     1|chevrolet chevell...|
|15.0|        8|       350.0|       165|  3693|        11.5|  70|     1|   buick skylark 320|
|18.0|        8|       318.0|       150|  3436|        11.0|  70|     1|  plymouth satellite|
|16.0|        8|       304.0|       150|  3433|        12.0|  70|     1|       amc rebel sst|
|17.0|        8|       302.0|       140|  3449|        10.5|  70|     1|         ford torino|
|15.0|        8|       429.0|       198|  4341|        10.0|  70|     1|    ford galaxie 500|
|14.0|        8|       454.0|       220|  4354|         9.0|  70|     1|    chevrolet impala|
|14.0|        8|       440.0|       215|  4312|         8.5|  70|     1|   plymouth fury iii|
|14.0|        8|       455.0|       225|  4425|        10.0|  70|     1|    pontiac catalina|
|15.0|        8|       390.0|       190|  3850|         8.5|  70|     1|  amc ambassador dpl|
|15.0|        8|       383.0|       170|  3563|        10.0|  70|     1| dodge challenger se|
|14.0|        8|       340.0|       160|  3609|         8.0|  70|     1|  plymouth 'cuda 340|
|15.0|        8|       400.0|       150|  3761|         9.5|  70|     1|chevrolet monte c...|
|14.0|        8|       455.0|       225|  3086|        10.0|  70|     1|buick estate wago...|
|24.0|        4|       113.0|        95|  2372|        15.0|  70|     3|toyota corona mar...|
|22.0|        6|       198.0|        95|  2833|        15.5|  70|     1|     plymouth duster|
|18.0|        6|       199.0|        97|  2774|        15.5|  70|     1|          amc hornet|
|21.0|        6|       200.0|        85|  2587|        16.0|  70|     1|       ford maverick|
|27.0|        4|        97.0|        88|  2130|        14.5|  70|     3|        datsun pl510|
|26.0|        4|        97.0|        46|  1835|        20.5|  70|     2|volkswagen 1131 d...|

only showing top 20 rows`

If everything loaded correctly, we can move on to data cleaning and feature engineering.
## 3. Data Cleaning 
Now, we are going to clean the data that we know has some problems, and perform a little feature engineering to turn our strings into indexed items for our random forest model pipeline.

If everything loaded correctly, you might notice that the `'horsepower'` column, which should be numeric, is showing up as a `StringType`. Why is this? Well, there are some `"?"` entries, so the file was read in as containing strings. We'll want to filter those rows out and recast the column as integers (`IntegerType`).

 **1. Fix the Horsepower Column**
`// Remove rows from dataframe that have "?" entries in the 'horsepower' column:
df = df.filter("horsepower != '?'")`

`// Cast the 'horsepower' column as IntegerType
df = df.withColumn("horsepower", df("horsepower").cast(IntegerType))`

Take a look at the data again to make sure things look right.

`df.show()`
**2. Fixing Duplicate Spellings with regexp_replace**
Look at the `"name"` column and select a few to do some regex experiments. We think acceleration is going to be similar for similar makes of automobiles, so we just want the first word from `"name"`. We can extract this using regex within Spark's `regexp_extract` and `regexp_replace` packages.

`// Use a regular expression code to extract the first word from the "name" string. // Create a new column, named "make"
df = df.withColumn("make", regexp_extract($"name", "^\\w+", 0))`

Once we get the makes of the automobiles into a separate column, we need to handle the cases in which the make has been misspelled. If you call `df.groupBy("make").show()`, you will see that "chevrolet" has also been entered into the dataframe as "chevy" and "chevroelt", so we need to use the `regexp_replace` to replace the various versions with only one (hopefully, the correct) spelling. You can look at the code and see that there were other duplicates that needed to be replaced:

`// Examining the data -- shows that there are multiple variations of some car makes:
df = df.withColumn("make", regexp_replace($"make", "(chevy|chevroelt)", "chevrolet"))
df = df.withColumn("make", regexp_replace($"make", "capri", "ford"))
df = df.withColumn("make", regexp_replace($"make", "hi", "ih"))
df = df.withColumn("make", regexp_replace($"make", "maxda", "mazda"))
df = df.withColumn("make", regexp_replace($"make", "toyouta", "toyota"))
df = df.withColumn("make", regexp_replace($"make", "vokswagen", "volkswagen"))
df.groupBy("make").count().orderBy($"make").show(300)`

**3. Feature Engineering with StringIndexer and OneHotEncoder**
For the `StringIndexer`, we will create a separate `StringIndexer` object for each `StringType` column. There are two cases:
<ul>
	<li>the column contains strings, but they do not represent categories, </li>
	<li>the column contains strings that do represent categories.</li>
</ul>
In the first case, we will only apply the `StringIndexer` and then pass the indexed column into a pipeline. In the second case, we will then apply a`OneHotEncoder` to the indexed column so that the model knows that the feature is categorical. This is the case we have with our `"make"` column. 

* First, create a `StringIndexer` object and assign it to `makeIndexer`. The chained commands needed for this method to work are:
<ol>
	<li>`.setInputCol("stringColumnName")`</li>
	<li>`.setOutputCol("columnNameIndexed")`</li>
	<li> OPTIONAL: `.setHandleInvalid("keep")`</li>
</ol>
The first input is the column name that needs to be transformed, which in our case is `"make"`. The second input is the name you want the indexer object to output, once the column has been indexed. Good practice is to make the new indexed column name according to following structure:

`"originalColumnName" + "Indexed"`

In our case, we want to set the output column name to `"makeIndexed"` because the original column is named `"make"`. Using a pipeline, we won't be explicitly referencing the output of the Indexer and Encoder objects, but it's good to understand what the output will be.

`// Feature engineering on car "make" with StringIndexer 
val makeIndexer = new StringIndexer()
.setInputCol("make")
.setOutputCol("makeIndexed")
.setHandleInvalid("keep")`

Once the `"make"` column has been indexed, we will use `OneHotEncoder` to create an encoded object, which we will call `"makeEncoded"` using the same logic described above.

`// Feature engineering on "makeIndexed" using OneHotEncoder
val makeEncoder = new OneHotEncoder()
.setInputCol(makeIndexer.getOutputCol)
.setOutputCol("makeEncoded")`

You will notice that the input column for the `OneHotEncoder` was a call to the output column of the `makeIndexer` object. 

**4. Create a Vector of Features to Input into the Model**
The rest of our features are `DoubleType` or `IntegerType`, so we don't have any more transformations to make on our data before transforming it into a feature vector. We need to gather all the feature columns from our dataframe together and pass them through a `VectorAssembler` object, which will transform them from their dataframe shape of columns and rows into an array of rows, each of which is an array of features.

* **This is a big difference between scikit-learn and Spark:** Spark machine learning models take in just two elements: 
<ul>
	<li>"label"</li>
	<li>"features"</li>
</ul>
`"Label"` is your target column,in this case `"acceleration"`. `"Features"` is an array of all your features. Remember: Spark models do not take dataframes as inputs, so use `VectorAssembler` to create a vector of features from your dataframe.

`// Create a vector of model features: "features"
val assembler = new VectorAssembler()
   .setInputCols(Array("mpg", "cylinders", "displacement", "horsepower", "weight","year", "origin",makeEncoder.getOutputCol))
   .setOutputCol("features")`

Great! We have already imported our data and Spark pacakges, we've cleaned up the data and designed some feature engineering, and we've set up the model's "label" and "features" arrays. We are ready to build our Random Forest model with a Pipeline!

## Splitting Data Into Training and Testing Data

At this point, we haven't split our data into a training data and test data. **We must do this before letting the model fit any of our data** to prevent data leakage into our model. We don't want our model to learn from our test data (i.e. out-of-sample data) because we need to use it test the accuracy of our model. For example, in a business case in which you have only a small amount of data in which to both set up your model and test your model's accuracy, your test data must be kept separate from the entire process of model training / learning so that you have the opportunity to prevent overfitting before your model gets launched into the real world where mistakes are costly in terms of clients and profitability.

So, let's split our dataframe into training data and testing data with an 80/20 split. In scikit-learn, you would recognize this as the `train_test_split()` method. In Spark, we use the `.randomSplit()` method that takes in two parameters:
<ol>
	<li>An array that designates the ratio you want for the train/test split, such as `Array(0.8,0.2)` or `Array(0.6, 0.4)`.</li>
	<li>A random seed that you can use to ensure your results can be replicated by yourself and others. </li>
</ol>

`// Implement a train test split of the dataframe with 80/20 split and seed of 2: train, test
val Array(train, test) = df.randomSplit(Array(.8,.2),2)`

## Setting up a RandomForestRegressor Model in Spark

The Spark random forest package is called `RandomForestRegressor`, and like all models we use in Spark, it expects only two arrays as input: `"label"` and `"features"`.

We can either recode the name of the target column of our dataframe as `"label"` or we can pass the real column name, `"acceleration"` into our model. The difference is that when we recode the column so that it has the name `"label"`, we don't have to pass it into our `RandomForestRegressor` model, as a parameter in `.setLabelCol()`. The model expects to find `"label"` by default, unless you override it by assigning a different column name. In this practice example, I am going to pass the real column target name, `"accelerator"` into the model.

Remember, we assembled all our feature columns into an array named `"features"` using `VectorAssembler()`. So, once `"features"` is created, we don't have to pass it explicitly into the model's `.setInputCols()`, because the model expects a `"features"` input column by default.

`// Instantiate a Random Forest Model
val rfr = new RandomForestRegressor()
  .setLabelCol("acceleration")`

## Setting up a Spark Pipeline

In Spark, you will create a new object that calls a `new Pipeline()` with all of its elements. Pipelines take two objects: transformers and estimators. We previously decided to transform the `"make"` column of our dataframe into a `StringIndexer` object named `makeIndexer`, and then convert that into a `OneHotEncoder` object, named `makeEncoder`. We also create a "features" `VectorAssembler` object called `assembler`. Now, as we set the stages of our pipeline, we'll put all of the transformers into the pipeline, along with the random forest model object we created, `rfr`. **Note: Make sure to call the `makeIndexer` before the `makeEncoder` in the pipeline, since the encoder function takes in the output of the indexer function as a parameter.**

`// Instantiate the Pipeline() with the "make", the features, and the model: pipeline
val pipeline = new Pipeline().setStages(Array(makeIndexer, makeEncoder, assembler, rfr))

// Train the model on training data with the pipeline.fit(): model
val model = pipeline.fit(train)

// Make predictions using model.transform(): predictions
val predictions = model.transform(test)

// Evaluate the model with an R-squared metric that takes in 
// the label and the predictions from predictions: evaluator 
val evaluator = new RegressionEvaluator()
  .setLabelCol("acceleration")
  .setMetricName("r2")`

## Grid Search
Now that we have our model and pipeline set up, we can conduct a grid search. In scikit-learn, I would use `GridSearchCV`, but for Spark, we imported and will use the package for `ParamGridBuilder`. 

The `ParamGridBuilder` takes in two types of parameters for a random forest:
<ul>
	<li>A parameter range for number of trees and max depth</li>
	<li>A build method</li>
</ul>

`// Create a Grid Search object for RandomForestRegressor: paramGrid
var paramGrid = new ParamGridBuilder()
  .addGrid(rfr.numTrees, Array(20, 30, 40))
  .addGrid(rfr.maxDepth, Array(1, 2, 3, 4, 5))
  .build()`

We also imported the Spark `CrossValidator` package to run cross validation on our pipeline as it fits trees across the (3 X 15) grid of parameters. 

`// Set up the CrossValidator and assign it to the object 'cv': cv
val cv = new CrossValidator()
  .setEstimator(pipe)
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(evaluator)
  .setNumFolds(3)

// Createa a cross validation object to fit on the training data: cvModel
val cvModel = cv.fit(train)

// Create an evaluate object to calculate the R-squared statistic on our test data: rsquaredCV
val rsquaredCV = evaluator.evaluate(cvModel.transform(test))

// Print the R-squared 
println("R-squared on test data = " + rsquaredCV)

res: rsquared_cv: Double = 0.7624816156777308`

Before the grid search, the best score was 0.64, and after it rose to 0.762. We found a nice boost in model accuracy by fitting different sizes of trees with a grid search. 

Now, you are going to want to know what the parameters were for the best model discovered by the grid search. 

You are going to search the output of the `.getEstimatorParamMaps` call on our cross validation object to find both the best score, and the parameter that it was associated with in the grid search. You can do this by zipping together the `cvModel` and the `.avgMetrics`, then finding the maximum `avgMetrics` by taking the max of the second item of the tuple `(_._2)`, and extracting the parameter for that value by accessing the first item in the `tuple (._1)`. Don't worry if it seems a bit difficult at first to chain and index items this way in Spark / Scala.

`// Get best parameters for model:
cvModel.getEstimatorParamMaps
  .zip(cvModel.avgMetrics)
  .maxBy(_._2)
  ._1<`
  
`res33: org.apache.spark.ml.param.ParamMap =
{
	rfr_6c28bd2d975a-maxDepth: 5,
	rfr_6c28bd2d975a-numTrees: 40
}`

We find that the best parameters were maxDepth of 5, and 40 trees. There is no "right" amount of tuning for your model. Play with the parameters and consider other features you could engineer, such as the model of the car.

I hope you enjoyed this practice example of Random Forests and Pipelines in Spark!
