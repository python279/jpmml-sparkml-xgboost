
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
import org.apache.spark.sql.functions.udf

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
val formula = new RFormula().setFormula("Adjusted ~ .").setFeaturesCol("features").setLabelCol("label").setHandleInvalid("keep")
val classifier = new XGBoostClassifier(Map("max_depth" -> 2, "objective" -> "binary:logistic", "num_round" -> 101, "num_workers" -> 2, "allow_non_zero_for_missing" -> "true", "missing" -> Float.NaN)).setLabelCol(formula.getLabelCol).setFeaturesCol(formula.getFeaturesCol)
val pipeline = new Pipeline().setStages(Array(formula, classifier))
val pipelineModel = pipeline.fit(df)
pipelineModel.write.overwrite.save("pipeline/XGBoostAudit")

val handleMissing = udf({x: String => {
  if (x == null) {
    "__unknown"
  } else {
    x
  }
}})
df = {
  for(field <- fields.filter(field => field.dataType.isInstanceOf[StringType])){
    df = df.withColumn(field.name, handleMissing(df(field.name)))
  }
  df
}
var precision = 1e-1
var zeroThreshold = 1e-1
val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).verify(df, precision, zeroThreshold).buildByteArray()
Files.write(Paths.get("pmml/XGBoostAudit.pmml"), pmmlBytes)
