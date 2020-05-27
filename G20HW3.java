import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.util.*;

public class G20HW3 {

    public static void main(String[] args) throws IOException {

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path num_diversity num_partitions");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        int K = Integer.parseInt(args[1]); //parameter for diversity maximization
        int L = Integer.parseInt(args[2]); // number of partitions
        String inputPath = args[0]; //file path

        //reading the points in a JavaRDD
        JavaRDD<Vector> inputPoints = sc.textFile(inputPath).map((str)->{
            String[] tokens = str.split(",");
            double[] data = new double[tokens.length];
            for (int i=0; i<tokens.length; i++) {
                data[i] = Double.parseDouble(tokens[i]);
            }
            return Vectors.dense(data);
        }).repartition(L).cache();
        long numdocs =  inputPoints.count(); //force the loading for avoiding lazy evaluation

    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

    /*implements the 4-approximation MapReduce algorithm for diversity maximization
    * Round 1: subdivides pointsRDD into L partitions and extracts k points from each partition using
    * the Farthest-First Traversal algorithm.
    * Hints :  • Recycle the implementation of FFT algorithm developed for HW 2;
    * • For the partitioning, invoking repartition(L) when the RDD was created, we can use the Spark Partitions,
    * accessing them through the mapPartition method.
    * Round 2: collects the L*k points extracted in Round 1 from the partitions into a set called coreset and returns
    * the k points computed by runSequential(coreset,k)
    * */

    public static void runMapReduce(JavaRDD<Double> pointsRDD,int k, int L){}

    //receives in input a set of points (pointSet) and computes the average distance between all pairs of points.
    public static Double measure(ArrayList<Vector> pointsSet){
        return null;
    }

}
