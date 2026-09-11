# Prueba - Modelo predictivo en Apache Spark MLlib
# Clasificación binaria de transacciones riesgosas
# Ejecutar en Google Colab, Databricks o entorno con PySpark.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ------------------------------------------------------------
# 1. Sesión Spark
# ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("PruebaSparkMLlibVentas")
    .getOrCreate()
)

RUTA = "ventas_simuladas_mllib.csv"

# ------------------------------------------------------------
# 2. Carga y validación inicial
# ------------------------------------------------------------
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(RUTA)
)

print("=== ESQUEMA INICIAL ===")
df.printSchema()

print("=== MUESTRA INICIAL ===")
df.show(10, truncate=False)

# Aseguramos tipos correctos.
df = (
    df
    .withColumn("Cantidad", F.col("Cantidad").cast(IntegerType()))
    .withColumn("Precio_Unitario", F.col("Precio_Unitario").cast(DoubleType()))
    .withColumn("Monto_Total", F.col("Monto_Total").cast(DoubleType()))
    .withColumn("Hora", F.col("Hora").cast(IntegerType()))
    .withColumn("label", F.col("label").cast(DoubleType()))
    .dropna(subset=[
        "Sucursal", "Producto", "Cantidad",
        "Precio_Unitario", "Monto_Total", "Hora", "label"
    ])
)

print("=== VALIDACIÓN DEL DATASET ===")
df.printSchema()
df.select(
    "Sucursal","Producto","Cantidad","Precio_Unitario",
    "Monto_Total","Hora","label"
).show(10, truncate=False)

print("=== DISTRIBUCIÓN DEL LABEL ===")
df.groupBy("label").count().orderBy("label").show()

# ------------------------------------------------------------
# 3. Preparación de features con Pipeline
# ------------------------------------------------------------
# StringIndexer transforma categorías en índices.
index_sucursal = StringIndexer(
    inputCol="Sucursal",
    outputCol="Sucursal_idx",
    handleInvalid="keep"
)

index_producto = StringIndexer(
    inputCol="Producto",
    outputCol="Producto_idx",
    handleInvalid="keep"
)

# OneHotEncoder evita interpretar el índice categórico como una magnitud.
encoder = OneHotEncoder(
    inputCols=["Sucursal_idx","Producto_idx"],
    outputCols=["Sucursal_ohe","Producto_ohe"]
)

# Ensamblaje preliminar de variables.
assembler = VectorAssembler(
    inputCols=[
        "Sucursal_ohe",
        "Producto_ohe",
        "Cantidad",
        "Precio_Unitario",
        "Monto_Total",
        "Hora"
    ],
    outputCol="features_raw",
    handleInvalid="keep"
)

# Estandarización de variables numéricas/ensambladas.
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

# Logistic Regression para clasificación binaria.
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.05,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=[
    index_sucursal,
    index_producto,
    encoder,
    assembler,
    scaler,
    lr
])

# ------------------------------------------------------------
# 4. División entrenamiento / prueba
# ------------------------------------------------------------
train_df, test_df = df.randomSplit([0.75, 0.25], seed=42)

# Cache opcional para reutilización.
train_df = train_df.cache()
test_df = test_df.cache()

print("Train:", train_df.count())
print("Test :", test_df.count())

# ------------------------------------------------------------
# 5. Entrenamiento y predicción
# ------------------------------------------------------------
modelo = pipeline.fit(train_df)
predicciones = modelo.transform(test_df)

print("=== EVIDENCIA FEATURES + LABEL ===")
predicciones.select("features","label").show(10, truncate=False)

print("=== TABLA DE PREDICCIONES ===")
predicciones.select(
    "label","prediction","probability",
    "Sucursal","Producto","Monto_Total","Hora"
).show(30, truncate=False)

# ------------------------------------------------------------
# 6. Evaluación
# ------------------------------------------------------------
eval_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

eval_accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

eval_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

auc = eval_auc.evaluate(predicciones)
accuracy = eval_accuracy.evaluate(predicciones)
f1 = eval_f1.evaluate(predicciones)

print(f"Accuracy    : {accuracy:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"AreaUnderROC: {auc:.4f}")

# Matriz de confusión simple con Spark.
print("=== MATRIZ DE CONFUSIÓN ===")
predicciones.groupBy("label","prediction").count().orderBy("label","prediction").show()

# ------------------------------------------------------------
# 7. Persistencia opcional del Pipeline entrenado
# ------------------------------------------------------------
# Descomentar en un entorno persistente:
# modelo.write().overwrite().save("modelo_riesgo_transacciones_pipeline")

# ------------------------------------------------------------
# 8. Posibles mejoras futuras
# ------------------------------------------------------------
# - CrossValidator o TrainValidationSplit.
# - ParamGridBuilder para regParam / elasticNetParam.
# - RandomForestClassifier y comparación con LogisticRegression.
# - Nuevas features: día de semana, promociones, clima, inventario,
#   historial del producto/sucursal y ventanas temporales.
# - Técnicas para desbalance si la proporción de riesgos cambia.
# - Persistencia del pipeline y scoring de nuevas transacciones.

spark.stop()
