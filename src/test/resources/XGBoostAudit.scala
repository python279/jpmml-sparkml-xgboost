import java.nio.file.{Files, Paths}
import java.io._
import java.io.{FileReader, BufferedReader}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, Imputer, VectorAssembler, Binarizer, RFormula}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{IntegerType, DoubleType, StringType, DataType, StructField, StructType, BooleanType}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.sparkml.model.HasRegressionTableOptions

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Audit.csv")
val schema = {
  val is = new FileReader("schema/Audit.json")
  val br = new BufferedReader(is)
  val schema = DataType.fromJson(br.readLine())
  schema.asInstanceOf[StructType]
}
val fields = schema.fields
df = {
  for(field <- fields){
    val column = df.apply(field.name).cast(field.dataType);
    df = df.withColumn("tmp_" + field.name, column).drop(field.name).withColumnRenamed("tmp_" + field.name, field.name)
  }
  df
}
val formula = new RFormula().setFormula("Adjusted ~ .").setFeaturesCol("features").setLabelCol("label")
val classifier = new XGBoostClassifier(Map("objective" -> "binary:logistic", "num_round" -> 10, "missing" -> 0.0, "allow_non_zero_missing" -> "true")).setLabelCol(formula.getLabelCol).setFeaturesCol(formula.getFeaturesCol)
val pipeline = new Pipeline().setStages(Array(formula, classifier))
val pipelineModel = pipeline.fit(df)
pipelineModel.write.overwrite.save("pipeline/XGBoostAudit")

//var xgbDf = pipelineModel.transform(df)
//val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index).toFloat }
//xgbDf = xgbDf.selectExpr("prediction as Adjusted", "probability")
//xgbDf = xgbDf.withColumn("AdjustedTmp", xgbDf("Adjusted").cast(IntegerType).cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")
//xgbDf = xgbDf.withColumn("probability(0)", vectorToColumn(xgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(xgbDf("probability"), lit(1))).drop("probability")
//xgbDf.coalesce(1).write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").save("csv/XGBoostAudit.csv")

var precision = 1e-14
var zeroThreshold = 1e-14
val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).verify(df, precision, zeroThreshold).buildByteArray()
Files.write(Paths.get("pmml/XGBoostAudit.pmml"), pmmlBytes)
