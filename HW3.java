import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.storage.StorageLevel;
import scala.Array;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import static java.lang.Long.parseLong;

public class HW3 {
    final int SEED = 1237030;
    static long maximumSize = 0;
    public static void main(String[] args) throws IOException {

        System.setProperty("hadoop.home.dir", "C:\\UNIPD\\big_data");

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: num_partitions subsets file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read number of partitions
        final int K = Integer.parseInt(args[0]);
        final int L = Integer.parseInt(args[1]);

        //Start time
        long startTime = System.currentTimeMillis();

        // Read input file and subdivide it into K random partitions
        JavaRDD<Vector> inputPoints = sc.textFile(args[2]).map(HW3::strToVector).repartition(L).cache();
        inputPoints.count();

        //End time
        long endTime = System.currentTimeMillis();

        //Calculate the number of inputs
        long numberInput = inputPoints.count();

        //Debug's prints
        System.out.println("Number of points = " + numberInput);
        System.out.println("K = " + K);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (endTime-startTime)+"ms\n");

        ArrayList<Vector> solution = runMapReduce(inputPoints, K, L);
        double averageDistance = measure(solution);
        System.out.println("Average distance = " + averageDistance);
    }

    /**
     * Implements of the 4-approximation MapReduce algorithm for diversity maximization
     * @param pointsRDD The points that are analyzed
     * @param K Number of partitions
     * @param L Number of subsets
     * @return An arrayList containing the coreset of points
     */
    static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, final int K, final int L){
        long startTime = System.currentTimeMillis();
        //Round 1
        List<List<Vector>> coresetList = pointsRDD.mapPartitions((vector) ->{
            ArrayList<Vector> arrayList = new ArrayList<>();
            while(vector.hasNext()){
                arrayList.add(vector.next());
            }
            return farthestFirstTraversal(arrayList, K).iterator();
        })
        .glom().collect();
        int size = coresetList.size();
        long endTime = System.currentTimeMillis();
        System.out.println("Runtime of Round 1 = " + (endTime-startTime)+"ms");

        startTime = System.currentTimeMillis();
        //Round 2
        ArrayList<Vector> coreset = new ArrayList<>();
        for(List<Vector> listVector : coresetList){
            coreset.addAll(listVector);
        }
        ArrayList<Vector> selectedPoints = runSequential(coreset, K);
        size = selectedPoints.size();
        endTime = System.currentTimeMillis();
        System.out.println("Runtime of Round 2 = " + (endTime-startTime)+"ms\n");
        return selectedPoints;
    }

    /**
     * Receives in input a set of points and computes the average distance between all pairs of points
     * @param pointSet The initial set of points
     * @return the average distance between all pairs of points
     */
    static double measure(ArrayList<Vector> pointSet){
        double dimension = pointSet.size();
        double sumOfDistances = 0;
        for(int i = 0; i < dimension; i++){
            for(int j = i + 1; j < dimension; j++ ){
                sumOfDistances += Math.sqrt(Vectors.sqdist(pointSet.get(i), pointSet.get(j)));
            }
        }
        return (sumOfDistances/dimension);
    }

    public static ArrayList<Vector> farthestFirstTraversal(ArrayList<Vector> S, int k) throws IOException{
        if(k>=S.size()){
            throw new IllegalArgumentException("Integer k greater than the cardinality of input set");
        }
        Random rand = new Random();
        rand.setSeed(1237030);
        //array of the distances of each point to the closest center
        ArrayList<Double> minDists = new ArrayList<>(S.size());
        ArrayList<Vector> centers = new ArrayList<>();
        //put random point in centers
        centers.add( S.get(rand.nextInt(S.size())) );
        double maxDist, dist;
        int idx=0;
        for(int j=0; j<k-1; j++) {
            // max distance from the current set of centers
            maxDist=0;
            for(int i=0; i<S.size(); i++) {
                //distance between
                dist=Vectors.sqdist(centers.get(j), S.get(i));
                //only for the first center fill minDists with the distances between it and every other point
                if(j==0) {
                    minDists.add(i, dist);
                    //for the other centers update minDists only if a smaller distance is found
                }else{
                    minDists.set( i, Math.min(minDists.get(i), dist) );
                }
                //if a bigger distance is found update the next center to the current point
                if (minDists.get(i) > maxDist) {
                    maxDist = minDists.get(i);
                    idx=i;
                }
            }
            centers.add(S.get(idx));
        }
        return centers;
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


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Conversion of string to a vector
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    public static org.apache.spark.mllib.linalg.Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }
}
