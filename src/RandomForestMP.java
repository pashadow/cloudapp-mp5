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
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.HashMap;
import java.util.regex.Pattern;

public final class RandomForestMP {

   private static class ParseTrainingData implements Function<String, LabeledPoint> {
        private static final Pattern SPACE = Pattern.compile(",");

        public LabeledPoint call(String line) {
            String[] tok = SPACE.split(line);
			double label = Double.parseDouble(tok[tok.length-1]);
			double[] point = new double[tok.length-1];
			for (int i = 0; i < tok.length - 1; ++i) {
				point[i] = Double.parseDouble(tok[i]);
			}
			return new LabeledPoint(label, Vectors.dense(point));
        }
    }

    private static class ParseTestData implements Function<String, Vector> {
        private static final Pattern SPACE = Pattern.compile(",");

        public Vector call(String line) {
            String[] tok = SPACE.split(line);
			double[] point = new double[tok.length-1];
            for (int i = 0; i < tok.length - 1; ++i) {
                point[i] = Double.parseDouble(tok[i]);
            }
            return Vectors.dense(point);			
        }
    }

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
		JavaRDD<LabeledPoint> trainingData = sc.textFile(training_data_path).map(new ParseTrainingData());
		JavaRDD<Vector> testData = sc.textFile(test_data_path).map(new ParseTestData());
		
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
		
/*		// Evaluate model on test instances and compute test error
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
*/
        JavaRDD<LabeledPoint> results = testData.map(new Function<Vector, LabeledPoint>() {
            public LabeledPoint call(Vector points) {
                return new LabeledPoint(model.predict(points), points);
            }
        });

        results.saveAsTextFile(results_path);

        sc.stop();
    }

}
