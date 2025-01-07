import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.fs.FileSystem;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Task3 {

    public static class CentroidMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private List<double[]> centroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // Load centroids from the local seed points file
            centroids = loadCentroids(context.getConfiguration().get("seedFilePath"));
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] point = value.toString().split(",");
            double x = Double.parseDouble(point[0]);
            double y = Double.parseDouble(point[1]);
            double z = Double.parseDouble(point[2]);
            int nearestCentroid = findNearestCentroid(x, y, z);
            context.write(new IntWritable(nearestCentroid), value);
        }

        private int findNearestCentroid(double x, double y, double z) {
            int nearestIndex = -1;
            double minDist = Double.MAX_VALUE;

            for (int i = 0; i < centroids.size(); i++) {
                double[] centroid = centroids.get(i);
                double distance = Math.sqrt(Math.pow(x - centroid[0], 2) + Math.pow(y - centroid[1], 2) + Math.pow(z - centroid[2], 2));

                if (distance < minDist) {
                    minDist = distance;
                    nearestIndex = i;
                }
            }

            return nearestIndex;
        }

        private List<double[]> loadCentroids(String seedFilePath) throws IOException {
            List<double[]> centroids = new ArrayList<>();
            FileSystem fs = FileSystem.getLocal(new Configuration());
            Path path = new Path(seedFilePath);
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));

            String line;
            while ((line = br.readLine()) != null) {
                String[] coordinates;
                // Check if the line contains a tab character (indicating an iteration output file)
                if (line.contains("\t")) {
                    // Split the line by tab to separate the index from the coordinates
                    String[] parts = line.split("\t");
                    if (parts.length == 2) {
                        coordinates = parts[1].split(",");
                    } else {
                        System.err.println("Invalid line format (expected index and coordinates): " + line);
                        continue;
                    }
                } else {
                    // For the initial seed file, directly split the line by commas
                    coordinates = line.split(",");
                }

                // Ensure we have exactly 3 coordinates (x, y, z)
                if (coordinates.length == 3) {
                    try {
                        double[] centroid = new double[3];
                        centroid[0] = Double.parseDouble(coordinates[0].trim());  // x value
                        centroid[1] = Double.parseDouble(coordinates[1].trim());  // y value
                        centroid[2] = Double.parseDouble(coordinates[2].trim());  // z value
                        centroids.add(centroid);
                    } catch (NumberFormatException e) {
                        System.err.println("Error parsing centroid values: " + line);
                    }
                } else {
                    System.err.println("Invalid centroid format: " + line);
                }
            }
            br.close();
            return centroids;
        }
    }

    public static class CentroidReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumX = 0, sumY = 0, sumZ = 0;
            int count = 0;
            for (Text value : values) {
                String[] point = value.toString().split(",");
                sumX += Double.parseDouble(point[0]);
                sumY += Double.parseDouble(point[1]);
                sumZ += Double.parseDouble(point[2]);
                count++;
            }
            double newX = sumX / count;
            double newY = sumY / count;
            double newZ = sumZ / count;
            context.write(key, new Text(newX + "," + newY + "," + newZ));
        }
    }

    // Method to calculate centroid displacement (used for convergence check)
    public static double calculateCentroidDisplacement(String prevSeedFile, String currSeedFile) throws IOException {
        List<double[]> prevCentroids = loadCentroidsFromFile(prevSeedFile);
        List<double[]> currCentroids = loadCentroidsFromFile(currSeedFile);

        double totalDisplacement = 0.0;
        for (int i = 0; i < prevCentroids.size(); i++) {
            double[] prev = prevCentroids.get(i);
            double[] curr = currCentroids.get(i);
            double displacement = Math.sqrt(Math.pow(curr[0] - prev[0], 2) + Math.pow(curr[1] - prev[1], 2) + Math.pow(curr[2] - prev[2], 2));
            totalDisplacement += displacement;
        }
        return totalDisplacement;
    }

    // Helper method to load centroids from file
    public static List<double[]> loadCentroidsFromFile(String seedFilePath) throws IOException {
        List<double[]> centroids = new ArrayList<>();
        FileSystem fs = FileSystem.getLocal(new Configuration());
        Path path = new Path(seedFilePath);
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));

        String line;
        while ((line = br.readLine()) != null) {
            String[] coordinates;
            if (line.contains("\t")) {
                String[] parts = line.split("\t");
                coordinates = parts[1].split(",");
            } else {
                coordinates = line.split(",");
            }

            if (coordinates.length == 3) {
                try {
                    double[] centroid = new double[3];
                    centroid[0] = Double.parseDouble(coordinates[0].trim());
                    centroid[1] = Double.parseDouble(coordinates[1].trim());
                    centroid[2] = Double.parseDouble(coordinates[2].trim());
                    centroids.add(centroid);
                } catch (NumberFormatException e) {
                    System.err.println("Error parsing centroid values: " + line);
                }
            }
        }
        br.close();
        return centroids;
    }

    public static void runIteration(Configuration conf, String inputPath, String seedFilePath, String outputPath, int iteration) throws Exception {
        conf.set("seedFilePath", seedFilePath);

        Job job = Job.getInstance(conf, "KMeans Iteration " + iteration);
        job.setJarByClass(Task3.class);
        job.setMapperClass(CentroidMapper.class);
        job.setReducerClass(CentroidReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath + "/iteration_" + iteration));

        if (!job.waitForCompletion(true)) {
            System.err.println("Iteration " + iteration + " failed.");
            System.exit(1);
        }
    }

    public static void main(String[] args) throws Exception {

        //Pass the correct file path in this format - "file:///C:/path/to/file/filename"

        String inputPath = ".../Project2/3d_points_dataset.csv";
        String seedFilePath = ".../Project2/seed_points_K5.csv";
        String outputBasePath = ".../Project2/output/task3";
        int numIterations = 30;
        double threshold = 5;  // Convergence threshold for centroid displacement

        Configuration conf = new Configuration();
        String prevSeedFile = seedFilePath;

        for (int i = 0; i < numIterations; i++) {
            runIteration(conf, inputPath, seedFilePath, outputBasePath, i);

            // Update the seed file path for the next iteration
            String currSeedFile = outputBasePath + "/iteration_" + i + "/part-r-00000";

            // Calculate the displacement of centroids
            double displacement = calculateCentroidDisplacement(prevSeedFile, currSeedFile);
            System.out.println("Centroid displacement after iteration " + i + ": " + displacement);

            // If the centroids have converged (displacement < threshold), stop early
            if (displacement < threshold) {
                System.out.println("Converged after iteration " + i + ". Terminating early.");
                break;
            }

            // Update the previous seed file for the next iteration
            prevSeedFile = currSeedFile;
            seedFilePath = currSeedFile;  // Use the current output as the new seed file
        }
    }
}
