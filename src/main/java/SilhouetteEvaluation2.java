import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.*;
import org.apache.hadoop.fs.FileSystem;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class SilhouetteEvaluation2 {

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
                    //System.out.println("Inside contains tab");
                    // Split the line by tab to separate the index from the coordinates
                    String[] parts = line.split("\t");
                    coordinates = parts[1].split(";")[0].split(","); // Get centroid coordinates

                    if (coordinates.length != 3) {
                        System.err.println("Invalid line format (expected index and coordinates): " + line);
                        continue;
                    }
                } else {
                    //System.out.println("Inside contains comma");
                    // For the initial seed file, directly split the line by commas
                    coordinates = line.split(",");
                }

                // Ensure we have exactly 3 coordinates (x, y, z)
                if (coordinates.length == 3) {
                    //System.out.println("coordinates length is 3");
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
                    //System.out.println("coordinates length is "+coordinates.length);
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

            // To store all points associated with this cluster
            StringBuilder points = new StringBuilder();

            for (Text value : values) {
                String[] point = value.toString().split(",");
                sumX += Double.parseDouble(point[0]);
                sumY += Double.parseDouble(point[1]);
                sumZ += Double.parseDouble(point[2]);
                count++;

                // Collect the point for output
                if (points.length() > 0) {
                    points.append("; "); // Delimiter between points
                }
                points.append(value.toString()); // Add the original point
            }

            // Calculate the new centroid
            double newX = sumX / count;
            double newY = sumY / count;
            double newZ = sumZ / count;

            // Create the output format: "centroid_x,centroid_y,centroid_z; point1; point2; ..."
            String outputValue = newX + "," + newY + "," + newZ + "; " + points.toString();

            // Write the new centroid and associated points to context
            context.write(key, new Text(outputValue));
        }
    }


    // Function to calculate Silhouette score
    private static void calculateSilhouetteScore(Configuration conf, String inputPath, String seedFilePath) throws Exception {
        //System.out.println("Calculating silhouette score...");

        // Load the points and their respective clusters from HDFS
        Map<Integer, List<double[]>> allClusters = new HashMap<>();
        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(inputPath);
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));

        String line;
        while ((line = br.readLine()) != null) {
            // Split the line by tab to separate cluster ID from the data
            String[] parts = line.split("\t");
            int clusterId = Integer.parseInt(parts[0].trim());
            //System.out.println("Cluster ID " + clusterId);

            // Handle the centroid part
            String[] centroidParts = parts[1].split(";")[0].split(","); // Get centroid coordinates
            double[] centroid = new double[3];

            try {
                centroid[0] = Double.parseDouble(centroidParts[0].trim());
                centroid[1] = Double.parseDouble(centroidParts[1].trim());
                centroid[2] = Double.parseDouble(centroidParts[2].trim());
            } catch (NumberFormatException e) {
                System.out.println("Error parsing centroid coordinates: " + e.getMessage());
                continue; // Skip this line if parsing fails
            }

            //System.out.println("Centroid for cluster " + clusterId + ": " + Arrays.toString(centroid));

            // Now handle the points after the centroid
            String[] pointsParts = parts[1].split(";"); // Split by semicolon to get points
            for (int i = 1; i < pointsParts.length; i++) { // Start from 1 to skip the centroid
                String pointStr = pointsParts[i].trim();
                String[] coordinates = pointStr.split(","); // Split the point by comma
                double[] point = new double[3];

                try {
                    point[0] = Double.parseDouble(coordinates[0].trim());
                    point[1] = Double.parseDouble(coordinates[1].trim());
                    point[2] = Double.parseDouble(coordinates[2].trim());
                    //System.out.println("Adding point to cluster " + clusterId + ": " + Arrays.toString(point));
                    allClusters.computeIfAbsent(clusterId, k -> new ArrayList<>()).add(point);
                } catch (NumberFormatException e) {
                    System.out.println("Error parsing point coordinates: " + e.getMessage());
                }
            }
        }
        br.close();

        // Debugging: Output loaded cluster points
        //System.out.println("Loaded cluster points:");
        //for (Map.Entry<Integer, List<double[]>> entry : allClusters.entrySet()) {
        //    System.out.println("Cluster " + entry.getKey() + " has " + entry.getValue().size() + " points.");
        //}

        // Compute Silhouette score for each cluster
        for (Map.Entry<Integer, List<double[]>> entry : allClusters.entrySet()) {
            int clusterId = entry.getKey();
            List<double[]> clusterPoints = entry.getValue();
            int count = clusterPoints.size();

            // Skip clusters with only one point
            if (count <= 1) {
                System.out.println("Cluster " + clusterId + " has too few points to calculate silhouette score.");
                continue;
            }

            // Calculate intra-cluster distance
            double totalIntraDistance = 0.0;
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < count; j++) {
                    if (i != j) {
                        double distance = Math.sqrt(Math.pow(clusterPoints.get(i)[0] - clusterPoints.get(j)[0], 2)
                                + Math.pow(clusterPoints.get(i)[1] - clusterPoints.get(j)[1], 2)
                                + Math.pow(clusterPoints.get(i)[2] - clusterPoints.get(j)[2], 2));
                        totalIntraDistance += distance;
                    }
                }
            }

            double averageIntraDistance = count > 1 ? totalIntraDistance / (count * (count - 1)) : 0;

            // Calculate inter-cluster distance
            double totalInterDistance = 0.0;
            int neighboringClusters = 0;

            for (Map.Entry<Integer, List<double[]>> otherEntry : allClusters.entrySet()) {
                if (otherEntry.getKey() != clusterId) {
                    neighboringClusters++;
                    for (double[] pointInCluster : clusterPoints) {
                        for (double[] neighborPoint : otherEntry.getValue()) {
                            double interDistance = Math.sqrt(Math.pow(pointInCluster[0] - neighborPoint[0], 2)
                                    + Math.pow(pointInCluster[1] - neighborPoint[1], 2)
                                    + Math.pow(pointInCluster[2] - neighborPoint[2], 2));
                            totalInterDistance += interDistance;
                        }
                    }
                }
            }

            double averageInterDistance = neighboringClusters > 0 ?
                    totalInterDistance / (count * neighboringClusters) : 0.0;

            // Compute Silhouette score
            if (averageIntraDistance > 0 || averageInterDistance > 0) {
                double silhouetteScore = (averageInterDistance - averageIntraDistance) /
                        Math.max(averageIntraDistance, averageInterDistance);
                System.out.println("Silhouette Score for Cluster " + clusterId + ": " + silhouetteScore);
            } else {
                System.out.println("Silhouette score cannot be computed for Cluster " + clusterId + " (zero intra/inter distances).");
            }
        }
    }


    public static void runIteration(Configuration conf, String inputPath, String seedFilePath, String outputPath, int iteration) throws Exception {
        conf.set("seedFilePath", seedFilePath);

        Job job = Job.getInstance(conf, "KMeans Iteration " + iteration);
        job.setJarByClass(SilhouetteEvaluation2.class);
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

        System.out.println("Iteration "+iteration);
        // Calculate Silhouette score after each iteration
        calculateSilhouetteScore(conf, outputPath + "/iteration_" + iteration + "/part-r-00000", seedFilePath);
        System.out.println();
    }

    public static void main(String[] args) throws Exception {

        //Pass the correct file path in this format - "file:///C:/path/to/file/filename"

        String inputPath = ".../Project2/3d_points_dataset.csv";
        String seedFilePath = ".../Project2/seed_points_K5.csv";
        String outputPath = ".../Project2/output/Silhouette2";
        int iterations = 5;  // Change iterations as needed

        Configuration conf = new Configuration();
        for (int i = 0; i < iterations; i++) {
            runIteration(conf, inputPath, seedFilePath, outputPath, i);

            // Update the seed file path for the next iteration
            seedFilePath = outputPath + "/iteration_" + i + "/part-r-00000";
        }
    }
}
