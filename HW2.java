import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;

public class HW2 {
    public static long finishTime;
    public static long startingTime;
    public static double maxDistance;
    public static int k;

    public static void main(String[] args){
        //Read the file from the name passed in args, if the file doesn't exist the program will exit
        String fileName = args[0];
        ArrayList<Vector> inputPoints = new ArrayList<>();
        try{
            inputPoints = readVectorsSeq(fileName);
        } catch (IOException error){
            System.out.println("Error -> 404 File Not Found");
            System.exit(1);
        }

        System.out.println("EXACT ALGORITHM");
        startingTime = System.currentTimeMillis();
        maxDistance = exactMPD(inputPoints);
        System.out.println("Max Distance = " + maxDistance);
        finishTime = System.currentTimeMillis();
        System.out.println("Running time = "+ (finishTime-startingTime) + " ms\n");

        System.out.println("2-APPROXIMATION ALGORITHM");
        startingTime = System.currentTimeMillis();
        k = inputPoints.size()/2;
        //Copy the arraylist
        ArrayList<Vector> copied = new ArrayList<>(inputPoints);
        System.out.println("k = " + k);
        maxDistance = twoApproxMPD(copied, k);;
        System.out.println("Max Distance = " + maxDistance);
        finishTime = System.currentTimeMillis();
        System.out.println("Running time = "+ (finishTime-startingTime) + " ms\n");

        System.out.println("k-CENTER-BASED ALGORITHM");
        startingTime = System.currentTimeMillis();
        k = 40;
        System.out.println("k = " + k);
        ArrayList<Vector> copied2 = new ArrayList<>(inputPoints);
        ArrayList<Vector> centers = kCenterMPD(copied2, k);
        maxDistance = exactMPD(centers);
        System.out.println("Max Distance = " + maxDistance);
        finishTime = System.currentTimeMillis();
        System.out.println("Running time = "+ (finishTime-startingTime) + " ms\n");
    }

    private static double exactMPD(ArrayList<Vector> inputPoints){
        //Variables of the method
        double maxDistance = 0;
        long finishTime;
        long startingTime = System.currentTimeMillis();

        for(int i = 0; i < inputPoints.size(); i++){
            for(int j = i+1; j < inputPoints.size(); j++){
                double currentDistance = Math.sqrt(Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
                if(currentDistance > maxDistance){
                    maxDistance = currentDistance;
                }
            }
        }
        finishTime = System.currentTimeMillis();

        //Print all the needed informations
        //System.out.println("Max Distance = " + maxDistance);
        //System.out.println("Running time = "+ (finishTime-startingTime) + " ms\n");
        return maxDistance;
    }

    private static double twoApproxMPD(ArrayList<Vector> inputPoints, int k){
        long finishTime;
        long startingTime = System.currentTimeMillis();
        double maxDistance = 0;
        Random randomizer = new Random();
        randomizer.setSeed(1237030);
        ArrayList<Vector> secondSet = new ArrayList<>();

        //Check if k is lower than the size of inputPoints
        if(k > inputPoints.size()){
            System.out.println("Error -> k is greater than the inputPoints's size");
            return -1;
        }

        while(secondSet.size() < k){
            secondSet.add(inputPoints.remove(randomizer.nextInt(inputPoints.size())));
        }

        for (Vector inputPoint : inputPoints) {
            for (Vector vector : secondSet) {
                double currentDistance = Math.sqrt(Vectors.sqdist(inputPoint, vector));
                if (currentDistance > maxDistance) {
                    maxDistance = currentDistance;
                }
            }
        }
        finishTime = System.currentTimeMillis();

        /*System.out.println("k = "+ k);
        System.out.println("Max Distance = " + maxDistance);
        System.out.println("Running time = "+ (finishTime-startingTime) + " ms\n");*/
        return maxDistance;
    }

    private static ArrayList<Vector> kCenterMPD(ArrayList<Vector> inputPoints, int k){
        long finishTime;
        long startingTime = System.currentTimeMillis();
        double maxDistance = 0;
        Random randomizer = new Random();
        randomizer.setSeed(1237030);

        //Check if k is lower than the size of inputPoints
        if(k > inputPoints.size()){
            System.out.println("Error -> k is greater than the inputPoints's size");
            return null;
        }

        //add a random point to the centers, when I add the center ti centers the selected point
        //is removed so the search of the new point on the set (inputPoints - centers) is easier
        ArrayList<Vector> centers = new ArrayList<>();
        centers.add(inputPoints.remove(randomizer.nextInt(inputPoints.size())));
        for(int i = 0; i < k-1; i++){
            maxDistance = 0;
            int maxDistanceIndex = 0;
            for(int j = 0; j < inputPoints.size(); j++){
                double currentDistance = Math.sqrt(Vectors.sqdist(inputPoints.get(j), centers.get(i)));
                if(currentDistance > maxDistance){
                    maxDistance = currentDistance;
                    maxDistanceIndex = j;
                }
            }
            centers.add(inputPoints.remove(maxDistanceIndex));
        }
        return centers;
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

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }
}
