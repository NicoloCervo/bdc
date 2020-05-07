import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import java.io.IOException;
import java.util.*;

import java.nio.file.Files;
import java.nio.file.Paths;

public class HW2 {
    public static void main(String[] args) throws IOException {

        String filename = args[0];
        ArrayList<Vector> inputPoints;

        //load file
        inputPoints = readVectorsSeq(filename);

        //int k = Integer.parseInt(args[1]);
        //System.out.println(k);

        //run methods and print times
        long startTime = System.nanoTime();
        //System.out.println(exactMPD(inputPoints));
        System.out.println("exactMDP time " + (System.nanoTime()-startTime)/1000000+" millisec");

        startTime = System.nanoTime();
        System.out.println(twoApproxMPD(inputPoints,5000));
        System.out.println("approxMDP time " + (System.nanoTime()-startTime)/1000000+" millisec");

        //twoApprox empties the array so read again, should pass a copy!!!
        inputPoints = readVectorsSeq(filename);

        startTime = System.nanoTime();
        System.out.println(kCenterMPD(inputPoints,5));
        System.out.println("kCenterMDP time " + (System.nanoTime()-startTime)/1000000+" millisec");

        System.out.println(exactMPD(kCenterMPD(inputPoints,5)));
    }

    public static double exactMPD(ArrayList<Vector> S){
        double maxDist=0;
        //try every possible pair and keep maximum distance found, I ran it only on small and aircraft, takes to long on the others
        for(int i=0;i<S.size();i++){
            for(int j=i;j<S.size();j++){
                double dist = Math.sqrt(Vectors.sqdist(S.get(i),S.get(j)));
                if(dist>maxDist){
                    maxDist=dist;
                }
            }
        }
        return  maxDist;
    }

    //try shuffling the array and using the first k elements as subset, should be faster
    public static double twoApproxMPD(ArrayList<Vector> S, int k) throws IOException{
        if(k>=S.size()){
            throw new IllegalArgumentException("Integer k greater than the cardinality of input set");
        }
        Random rand = new Random();
        rand.setSeed(1237541);
        double dist, maxDist=0;
        //move k vectors to subset
        ArrayList<Vector> subset = new ArrayList<>();
        for(int i=0;i<k;i++){
            subset.add(S.remove(rand.nextInt(S.size())));
        }
        //check distance between all pairs {(v1,v2): v1 € S, v2 € subset}
        for (Vector v1 : S) {
            for (Vector v2 : subset) {
                dist = Math.sqrt(Vectors.sqdist(v1, v2));
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }
        }
        return maxDist;
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
        //int idx;
        for(int j=0; j<k-1; j++) {
            maxDist=0;
            //add another point because i cant use ArrayLists
            centers.add(S.get(0));
            for(int i=0; i<S.size(); i++) {
                //distane between
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
                    //probably there is a better way to insert the new center in the ArrayList
                    //idx=i;
                    centers.remove(j+1);
                    centers.add(j+1, S.get(i));
                }
            }
            //centers.add(j+1, S.remove(idx));
        }
        return centers;
    }

    //Support methods

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
