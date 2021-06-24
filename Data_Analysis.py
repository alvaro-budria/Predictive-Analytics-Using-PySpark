"""

This script models the data with a Decision Tree Classifier. The steps are:

1. Load the data from the files that "Data_Management.py" has created, with function "load_data".
   This is achieved with the help of the variable "exec_time", which helps select the adequate directory of the data.
2. The dataset is split in two sets: training (70%) and testing (30%) and
3. a Decision Tree Classifier is trained with the training set. This is done by "train_model" function.
4. Finally, evaluation metrics for the model (accuracy, precision and recall) are calculated
   with the testing set. These metrics, as well as the model, are returned by the "run" function.
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# loads the data from the data file(s)
def load_data(spark, exec_time):
    data = ( spark.read.format("libsvm")
             .option("numFeatures", "4")
             .load("./data/{}/{}".format(exec_time, "part-*")) )
    return data


# Returns the testind dataset and the model obtained from the given data
def train_model(spark, data):
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 2 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.8, 0.2], seed=123)

    # Train a DecisionTree model.
    dec_tree = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dec_tree])

    # Train model
    return testData, pipeline.fit(trainingData)


# Computes testing accuracy, precision and recall
def metrics(spark, model, testData):
    # Make predictions
    predictions = model.transform(testData)

    positive_cases = predictions.filter(predictions["indexedLabel"] == 1.0)
    true_positive_cases = positive_cases.filter(positive_cases["prediction"] == 1.0)
    false_negative_cases = positive_cases.filter(positive_cases["prediction"] == 0.0)
    negative_cases = predictions.filter(predictions["indexedLabel"] == 0.0)
    true_negative_cases = negative_cases.filter(negative_cases["prediction"] == 0.0)
    false_positive_cases = negative_cases.filter(negative_cases["prediction"] == 1.0)

    #compute test accuracy
    accuracy = (true_positive_cases.count() + true_negative_cases.count()) / (predictions.count())
    print("Test Accuracy = %g " % (accuracy))

    precision = -99
    #compute test precision
    if (true_positive_cases.count() + false_positive_cases.count() > 0): # avoid division by zero
        precision = true_positive_cases.count() / (true_positive_cases.count() + false_positive_cases.count())
    print("Test Precision = %g " % (precision))

    recall = -99
    #compute test recall
    if (true_positive_cases.count() + false_negative_cases.count() > 0): # avoid division by zero
        recall = true_positive_cases.count() / (true_positive_cases.count() + false_negative_cases.count())
    print("Test Recall = %g " % (recall))

    return accuracy, precision, recall


def run(spark, exec_time):
    # Load the data stored in LIBSVM format as a DataFrame.
    data = load_data(spark, exec_time)

    # Train model
    [testData, model] = train_model(spark, data)

    # compute performance testing metrics
    [accuracy, precision, recall] = metrics(spark, model, testData)

    return [model, accuracy, precision, recall]
