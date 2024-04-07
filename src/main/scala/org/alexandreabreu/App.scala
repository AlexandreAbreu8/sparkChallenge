package org.alexandreabreu

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, collect_set, explode, isnan, row_number, to_timestamp, udf}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, functions}
import org.apache.spark.sql.types.{ArrayType, DoubleType, LongType, StringType, StructField, StructType, TimestampType}

import scala.language.postfixOps
import scala.util.matching.Regex

/**
 * @author ${user.name}
 */
object App {
  
  def main(args : Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("sparkChallenge")
      .master("local[*]")
      .config("spark.driver.bindAddress", "192.168.1.75")
      .getOrCreate()

    spark.conf.set("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")
    spark.conf.set("parquet.enable.summary-metadata", "false")
    spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    val appsDF = loadApps(spark)
    val reviewsDF = loadReviews(spark)

    val df_1:DataFrame = exercise1(reviewsDF)
    exercise2(appsDF)
    val df_3:DataFrame = exercise3(appsDF)
    val df_4 = exercise4(df_1, df_3)
    //Note: the pdf states to use df3 created in exercise 3 but that dataframe does not contain the average sentiment polarity
    //Therefore I'm using df4 instead
    val df_5:DataFrame = exercise5(df_4)
  }

  private def loadApps(session:SparkSession): DataFrame = {
    val customSchema = StructType(Seq(
      StructField("App", StringType, nullable = false),
      StructField("Category", StringType, nullable = true),
      StructField("Rating", StringType, nullable = true),
      StructField("Reviews", StringType, nullable = true),
      StructField("Size", StringType, nullable = true),
      StructField("Installs", StringType, nullable = true),
      StructField("Type", StringType, nullable = true),
      StructField("Price", StringType, nullable = true),
      StructField("Content Rating", StringType, nullable = true),
      StructField("Genres", StringType, nullable = true),
      StructField("Last Updated", StringType, nullable = true),
      StructField("Current Ver", StringType, nullable = true),
      StructField("Android Ver", StringType, nullable = true)
    ))

    session.read
      .option("header", value=true)
      .option("escape", "\"")
      .schema(customSchema)
      .csv("data/googleplaystore.csv")
  }

  private def loadReviews(session:SparkSession): DataFrame = {
    val customSchema = StructType(Seq(
      StructField("App", StringType, nullable = true),
      StructField("Translated_Review", StringType, nullable = true),
      StructField("Sentiment", StringType, nullable = true),
      StructField("Sentiment_Polarity", StringType, nullable = true),
      StructField("Sentiment_Subjectivity", StringType, nullable = true)
    ))

    session.read
      .option("header", value=true)
      .option("escape", "\"")
      .schema(customSchema)
      .csv("data/googleplaystore_user_reviews.csv")
  }

  private def exercise1(df:DataFrame): DataFrame = {
    df.groupBy("App")
      .agg(functions.avg("Sentiment_Polarity").as("Average_Sentiment_Polarity"))
      .na.fill(0)
  }

  private def exercise2(df:DataFrame): Unit = {
    val ex2DF:DataFrame = df.filter(df("Rating") >= "4.0" && df("Rating") =!= "NaN")
      .sort(df("Rating").desc)

    ex2DF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", value=true)
      .option("delimiter", "§")
      .csv("data/out/best_apps")
  }

  private def exercise3(df:DataFrame): DataFrame = {
    val removeDollar = udf((s: String) => s.replace("$", ""))
    val convertSize = udf((s: String) => {
      val mega = new Regex("^[0-9]+[.]?[0-9]+[M]$")
      val kilo = new Regex("^[0-9]+[.]?[0-9]+[k]$")
      if (mega.findAllIn(s).nonEmpty) {
        s.replace("M", "")
      } else if (kilo.findAllIn(s).nonEmpty) {
        (s.replace("k", "").toDouble*0.001).toString
      } else null
    })

    val df2 = df
      .withColumn("Rating", functions.when(isnan(col("Rating")), null).otherwise(col("Rating")).cast(DoubleType))
      .withColumn("Reviews", functions.when(isnan(col("Reviews")), "0").otherwise(col("Reviews")).cast(LongType))
      .withColumn("Size", convertSize(col("Size")).cast(DoubleType))
      .withColumn("Installs", functions.when(isnan(col("Installs")), null).otherwise(col("Installs")))
      .withColumn("Type", functions.when(isnan(col("Type")), null).otherwise(col("Type")))
      .withColumn("Price", removeDollar(col("Price")).cast(DoubleType)*0.9)
      .withColumn("Content_Rating", functions.when(isnan(col("Content Rating")), null).otherwise(col("Content Rating")))
      .withColumn("Genres", functions.split(col("Genres"), ";").cast(ArrayType(StringType)))
      .withColumn("Last_Updated", functions.when(isnan(col("Last Updated")), null).otherwise(to_timestamp(col("Last Updated"), "MMMM d, yyyy")).cast(TimestampType))
      .withColumn("Current_Version", functions.when(isnan(col("Current Ver")), null).otherwise(col("Current Ver")))
      .withColumn("Minimum_Android_Version", functions.when(isnan(col("Android Ver")), null).otherwise(col("Android Ver")))

    val window = Window.partitionBy(df2("App")).orderBy(df2("Reviews").desc)

    df2.show()
    val aux = df2
      .withColumn("line", row_number().over(window))
      .withColumn("Categories", collect_set(col("Category")).over(window))
      .filter(col("line") === 1)

    aux.select("App",
      "Categories",
      "Rating",
      "Reviews",
      "Size",
      "Installs",
      "Type",
      "Price",
      "Content_Rating",
      "Genres",
      "Last_Updated",
      "Current_Version",
      "Minimum_Android_Version")
  }

  private def exercise4(df1:DataFrame, df3:DataFrame): DataFrame = {
    val newDF = df3.join(df1, df3("App")===df1("App"), "left")
      .drop(df1("App"))

    newDF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", value=true)
      .option("compression", "GZIP")
      .parquet("data/out/googleplaystore_cleaned")

    newDF
  }

  private def exercise5(df:DataFrame): DataFrame = {
    val newDF = df.withColumn("Genre", explode(col("Genres")))
      .groupBy("Genre")
      .agg(
        functions.count("App").as("Count"),
        functions.avg("Rating").as("Average_Rating"),
        functions.avg("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity"))

    newDF.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", value=true)
      .option("compression", "GZIP")
      .parquet("data/out/googleplaystore_metrics")

    newDF
  }
}
