"""

This script predicts if a given aircraft is going to have an unscheduled maintenance,
based on its data from the specified date.

1. Function "gather_data" retrieves the data relating to the specified aircraft and date.
   In particular, it calculates its KPIs and its sensor value, by reading from the DW
   and from the .csv files.
2. With function "prediction", we use the model obtained from "Data_Analysis.py"
   to predict the future state of the aircraft.
   0: no unscheduled maintenance in the next seven days
   1: some unsheduled maintenance in the next seven days
3. The predicted value is returned.

"""

from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as f  # dayofmonth, month, year


# Retrieves the KPIs for aircraft A and date "date"
def read_DW(spark, A, date):
    username = "MyUsername"
    password = "MyPassword"

    DW = ( spark.read
    .format("jdbc")
    .option("driver","org.postgresql.Driver")
    .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
    .option("dbtable", "public.aircraftutilization")
    .option("user", username)
    .option("password", password)
    .load()
    .select( "aircraftid", "timeid", "flighthours",  # keep useful columns
             "flightcycles", "delayedminutes", "scheduledoutofservice") )

    # keep rows for aircraft A and "date" (with format ddmmyy)
    DW = ( DW
    .filter(DW.aircraftid == A)
    .filter(f.year(DW.timeid) == ("20" + date[4:6]))
    .filter(f.month(DW.timeid) == date[2:4])
    .filter(f.dayofmonth(DW.timeid) == date[0:2]) )

    return DW.select("flighthours", "flightcycles", "delayedminutes")


# Reads the sensor measurements from the .csv files for an aircraft "A"
# and a day "date", and returns their average
def read_sensor(spark, A, date):

    # select files of aircraft A
    sensor = ( spark.read
    .option("header", "True")
    .option("delimiter",";")
    .csv("./resources/trainingData/" + date + "*" + A + ".csv")
    .drop("series") )

    # Change type of column "value" to float
    sensor = ( sensor
    .withColumn("valueNum", sensor["value"].cast("float"))
    .drop("value") )

    avg = ( sensor
            .agg({"valueNum": "avg"})
            .withColumnRenamed("avg(valueNum)", "value") )
    return avg.select("value")


# gathers the necessary KPIs and sensor value for the specified aircraft "A" and date "date"
def gather_data(spark, A, date):
    sensor = read_sensor(spark, A, date)
    KPIs = read_DW(spark, A, date)
    new_ex = KPIs.join(sensor).collect()[0] # merge values in a single row

    # assemble dataframe in the adequate format
    df = spark.createDataFrame([
          (-1.0, Vectors.dense([new_ex.flighthours, new_ex.flightcycles,
                                new_ex.delayedminutes, new_ex.value]))], ["label", "features"])
    return df


# predicts whether the aircraft will have an unsheduled maintenance based on the valus in "df"
def prediction(spark, df, model):
    prediction = model.transform(df)
    maint = prediction.select("prediction").rdd.flatMap(list).collect()[0]
    return maint


def run(spark, model, A, date):
    df = gather_data(spark, A, date) # obtain the KPIs and sensor value
    pred = prediction(spark, df, model) # obtain final prediction: 0: "no maintenance", 1: "maintenance"
    return pred
