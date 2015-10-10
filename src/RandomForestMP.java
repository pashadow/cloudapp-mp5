import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.HashMap;
import java.util.regex.Pattern;

public final class RandomForestMP {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                    "Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		// Load and parse the data files
		JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(sc.sc(), training_data_path).toJavaRDD();
		JavaRDD<LabeledPoint> testData = MLUtils.loadLibSVMFile(sc.sc(), test_data_path).toJavaRDD();
		
		// Train a RandomForest model
        Integer numClasses = 2;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

		final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
		
		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel =
		  testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
			@Override
			public Tuple2<Double, Double> call(LabeledPoint p) {
			  return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
			}
		  });
		Double testErr =
		  1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			@Override
			public Boolean call(Tuple2<Double, Double> pl) {
			  return !pl._1().equals(pl._2());
			}
		  }).count() / testData.count();
		System.out.println("Test Error: " + testErr);
		System.out.println("Learned classification forest model:\n" + model.toDebugString());		

        JavaRDD<LabeledPoint> results = predictionAndLabel.map(new Function<Vector, LabeledPoint>() {
            public LabeledPoint call(Vector points) {
                return new LabeledPoint(model.predict(points), points);
            }
        });

        results.saveAsTextFile(results_path);
        sc.stop();
    }

}
