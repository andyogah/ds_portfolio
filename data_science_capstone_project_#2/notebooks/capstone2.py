# Databricks notebook source
## Predicting how much is being paid out to physicians based on huge datasets from Centers for Medicare and Medicaid Services and the US Food and Drug Administration using Gradient Boosted Trees 

# COMMAND ----------



# COMMAND ----------

df = spark.read.csv('/FileStore/tables/opaymnt_data.csv', header=True, inferSchema=True)

# COMMAND ----------

display(df.limit(5))

# COMMAND ----------

new_df = df.drop("Index", "Name_of_Associated_Covered_Drug_or_Biological1")

# COMMAND ----------

display(new_df.limit(5))

# COMMAND ----------

df2 = spark.read.csv('/FileStore/tables/opaymnt_data2.csv', header=True, inferSchema=True)

# COMMAND ----------

display(df2.limit(7))

# COMMAND ----------

new_df2 = df2.drop("Index")

# COMMAND ----------

display(new_df2.limit(5))

# COMMAND ----------

df3 = new_df.join(new_df2, ["First_and_Last_name", "Recipient_City", "Recipient_State"])
df3.cache()

# COMMAND ----------

display(df3)

# COMMAND ----------

df4 = df3.drop('npi', 'First_and_Last_name', 'Physician_Profile_ID')
df4.cache()

# COMMAND ----------

display(df4)

# COMMAND ----------

#This shows the number of records we have in the DataFrame
df4.count()

# COMMAND ----------

# This shows the number of columns and their names.
len(df4.columns), df4.columns

# COMMAND ----------

#take random sample of the dataset
df4_sample = df4.sample(False, 0.00005, None)

# COMMAND ----------

df4_sample.count()

# COMMAND ----------

#filling in the null values
df5 = df4_sample.fillna(-1)
display(df5)

# COMMAND ----------

df5.printSchema()

# COMMAND ----------

display(df5.describe())

# COMMAND ----------



# COMMAND ----------

##DATA PREPROCESSING

# COMMAND ----------

###One-Hot Encoding
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

# COMMAND ----------

categoricalColumns = ["Recipient_City", "Recipient_State", "Physician_Primary_Type", "specialty_description", "drug_name"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  # Use OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  # Add stages.  These are not run here, but will run all at once later on.
  stages += [stringIndexer, encoder]

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
#https://stackoverflow.com/questions/49167615/typeerror-unsupported-operand-types-for-map-and-float?rq=1
numericCols = ["Number_of_Payments_Included_in_Total_Amount", "bene_count", "total_claim_count", "total_day_supply", "total_drug_cost"]
assemblerInputs = list(map(lambda c: c + "classVec", categoricalColumns)) + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

# Create a Pipeline.
pipeline = Pipeline(stages=stages)
# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
#pipelineModel = pipeline.fit(df5)
#df6 = pipelineModel.transform(df5)
#OR 
df6 = pipeline.fit(df5).transform(df5)

# COMMAND ----------

display(df6)

# COMMAND ----------

# Keep relevant columns
selectedcols = ["Total_Amount_of_Payment_USDollars", "features"] + numericCols + ["Recipient_CityclassVec", "Recipient_StateclassVec", "Physician_Primary_TypeclassVec", "specialty_descriptionclassVec", "drug_nameclassVec"]
df7 = df6.select(selectedcols)
display(df7)

# COMMAND ----------



# COMMAND ----------

#SPLITTING DATA INTO TRAINING AND TEST SETS

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing.
#df7 = df6.drop("Recipient_City", "Recipient_State", "Physician_Primary_Type", "specialty_description", "drug_name")
train, test = df7.randomSplit([0.7, 0.3], seed = 100)
print("We have %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

display(train)

# COMMAND ----------



# COMMAND ----------

#RUNNING THE MACHINE LEARNING MODEL

# COMMAND ----------

train.printSchema()

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Takes the "features" column and learns to predict "cnt"


# COMMAND ----------

gbt = GBTRegressor(labelCol="Total_Amount_of_Payment_USDollars", featuresCol="features")
pipeline = Pipeline(stages=[gbt])

# COMMAND ----------

#rf = RandomForestRegressor(labelCol="Total_Amount_of_Payment_USDollars", featuresCol="features")
#pipeline = Pipeline(stages=[rf])

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [2, 5]).addGrid(gbt.maxIter, [10, 100]).build()
##paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 5]).addGrid(rf.maxIter, [10, 100]).build()

# COMMAND ----------

evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
#evaluator = RegressionEvaluator(metricName="rmse", labelCol=rf.getLabelCol(), predictionCol=rf.getPredictionCol())

# COMMAND ----------

crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator)

# COMMAND ----------

model = crossval.fit(train)

# COMMAND ----------

predictions = model.transform(test)

# COMMAND ----------

display(predictions.select("Total_Amount_of_Payment_USDollars", "prediction", "features"))

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#https://stackoverflow.com/questions/46372562/random-forest-regression-for-categorical-inputs-on-pyspark

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df10 = spark.read.csv('/FileStore/tables/med_test2.csv', header=True, inferSchema=True)

# COMMAND ----------

display(df10)

# COMMAND ----------

df20 = spark.read.csv('/FileStore/tables/x.csv', header=True, inferSchema=True)

# COMMAND ----------

display(df20)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType
#df3 = sqlContext.read.format("com.databricks.spark.csv").options(header='true', inferSchema='true').load('/FileStore/tables/yourdata.csv')