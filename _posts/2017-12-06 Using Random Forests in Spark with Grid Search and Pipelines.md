

** Overview
For this practice project, we are going to predict the acceleration of cars using data provided by Gareth James and Co at USC, found in the dataset `Auto.csv`, which can be downloaded at <a href="http://www-bcf.usc.edu/~gareth/ISL/data.html" target="_blank" rel="noopener">http://www-bcf.usc.edu/~gareth/ISL/data.html</a>.

A great place to get practice using Apache Spark and writing Scala scripts is on <a href="https://databricks.com/" target="_blank" rel="noopener">DataBricks</a>. I use a Scala notebook in this practice example.

You can sign up for the community edition, which is free.

<strong>Step 1: To sign up, visit the DataBricks site and sign up for an account:</strong>
<img class="alignnone size-full wp-image-47" src="https://robotallie.files.wordpress.com/2017/12/databrickshome1.jpeg" alt="DatabricksHome1" width="1223" height="520" />

 

<strong>Step 2: Start Today - Register with your contact information.</strong>

 

<img class="alignnone size-full wp-image-49" src="https://robotallie.files.wordpress.com/2017/12/databricks-step2.jpeg" alt="Databricks-Step2" width="1184" height="550" />

 

<strong>Step 3: After your have confirmed your account, on the Home dashboard of DataBricks you will select "DATA" to upload the Auto.csv dataset you downloaded from USC.</strong>

 

<img class="alignnone size-full wp-image-50" src="https://robotallie.files.wordpress.com/2017/12/databricksstep3.jpeg" alt="DataBricksStep3" width="468" height="544" />

 

<strong>Step 4: Either drag & drop the file (if you use Chrome) into the box to automatically upload it, or click on the box and search for the file in your file system.</strong>

 

<img class="alignnone size-full wp-image-51" src="https://robotallie.files.wordpress.com/2017/12/databricksstep5.jpeg" alt="DataBricksStep5" width="583" height="670" />

 

<strong>Step 5: Once the file has been uploaded to DataBricks, you will see a green checkmark above the file and the filepath to access the file when you want to load the data into your notebook. You must copy this filepath and save it for later.</strong>

 

<img class="alignnone size-full wp-image-52" src="https://robotallie.files.wordpress.com/2017/12/databricksstep5b.jpeg" alt="DataBricksStep5B" width="522" height="673" />

<strong>Step 6: You are ready to create a Scala notebook. Click on the "WORKSPACE" button on the left side-bar to take you to the dashboard of users and notebooks. </strong>

 

<img class="alignnone size-full wp-image-53" src="https://robotallie.files.wordpress.com/2017/12/databricksstep6.jpeg" alt="DataBricksStep6" width="522" height="673" />

<strong>Step 7: On the WORKSPACE dashboard</strong>:
<ol>
	<li>Click on "Users" on the Workspace tab.</li>
	<li>Then click on your "username" on the Users tab.</li>
	<li>Click on the "down arrow" on your username to get the drop-down list of options.</li>
	<li>Click on "Create" until the next drop-down menu opens.</li>
	<li>Choose "Scala"</li>
</ol>
<img class="alignnone size-full wp-image-54" src="https://robotallie.files.wordpress.com/2017/12/databricksstep7.jpeg" alt="DataBricksStep7" width="1078" height="562" />

 

Once you have completed this, you are ready to set up your data and Apache Spark package imports, which I cover in  <a href="https://robotallie.wordpress.com/2017/12/06/random-forests-with-pipelines-in-scala-part-2-import-the-scala-packages-and-dataset/" target="_blank" rel="noopener">Random Forests with Pipelines in Scala - Part 2: Import the Scala Packages and Dataset</a>

Other posts in this series:
<a href="https://robotallie.wordpress.com/2017/12/06/random-forests-with-pipelines-in-scala-part-3-cleaning-data-and-feature-engineering/">Part 3: Data Cleaning and Feature Engineering</a>
<a href="https://robotallie.wordpress.com/2017/12/06/random-forests-with-pipelines-in-scala-part-5-model-and-pipeline-assembly/">Part 4: Setting up the Model and Pipeline</a>
Part 5: Tuning and Evaluating the Model*