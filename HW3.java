import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import static java.lang.Long.parseLong;

public class HW3 {
    public static void main(String[] args) throws IOException {

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path K L");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read number of partitions
        int K = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);

        // Read input file and subdivide it into L random partitions
        JavaRDD<Vector> filePartitions = sc.textFile(args[0]).map(HW3::strToVector).repartition(L).cache();

        ArrayList<Vector> inputPoints = readVectorsSeq(args[0]);
        ArrayList<Vector> result = runSequential(inputPoints, K);
        System.out.println("sequential: "+ result+"\n avg dist: "+ avgDistance(result));

        result = runMapReduce(filePartitions, K, L);
        System.out.println("mapReduce:  " + result+"\n avg dist: "+ avgDistance(result));
    }

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

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L){
        System.out.println("num partitions: "+ pointsRDD.partitions().size());
        JavaRDD<Vector> centers = pointsRDD
            .mapPartitions((points)->{
                //move vectors in array
                ArrayList<Vector> pointsA = new ArrayList<>();
                while (points.hasNext())  pointsA.add(points.next());
                
                //return k centers found with Furthest First Traversal
                return kCenterMPD(pointsA, k).iterator();
            });
        ArrayList<Vector> coreset = new ArrayList<>();
        for(int i=0; i< centers.collect().toArray().length; i++){
            coreset.add(centers.collect().get(i));
        }
        return runSequential(coreset, k);
    }

    //Support methods

    public static float avgDistance(ArrayList<Vector> points){
        float distanceSum=0;
        int counter=0;
        for(int i=0; i< points.size(); i++){
            for(int j=i+1; j< points.size(); j++){
                distanceSum += Math.sqrt(Vectors.sqdist(points.get(i), points.get(j)));
                counter++;
            }
        }
        return distanceSum/counter;
    }

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) throws IOException{
        if(k>=S.size()){
            throw new IllegalArgumentException("Integer k greater than the cardinality of input set");
        }
        Random rand = new Random();
        rand.setSeed(1237541);
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

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

}