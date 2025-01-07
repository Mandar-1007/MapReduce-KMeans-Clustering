import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class SilhouetteEvaluation {

    // Load centroids from the seed file
    public static List<double[]> loadCentroids(String seedFilePath) throws IOException {
        List<double[]> centroids = new ArrayList<>();
        FileSystem fs = FileSystem.get(new Configuration());
        Path path = new Path(seedFilePath);
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));

        String line;
        while ((line = br.readLine()) != null) {
            String[] coordinates = line.split(",");
            double[] centroid = new double[3];
            centroid[0] = Double.parseDouble(coordinates[0].trim());
            centroid[1] = Double.parseDouble(coordinates[1].trim());
            centroid[2] = Double.parseDouble(coordinates[2].trim());
            centroids.add(centroid);
        }
        br.close();
        return centroids;
    }

    public static class SilhouetteMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private List<double[]> centroids;

        @Override
        protected void setup(Context context) throws IOException {
            String seedFilePath = context.getConfiguration().get("seedFilePath");
            centroids = loadCentroids(seedFilePath);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] point = value.toString().split(",");
            double x = Double.parseDouble(point[0]);
            double y = Double.parseDouble(point[1]);
            double z = Double.parseDouble(point[2]);

            // Find the nearest centroid
            int nearestCentroid = -1;
            double minDist = Double.MAX_VALUE;

            for (int i = 0; i < centroids.size(); i++) {
                double[] centroid = centroids.get(i);
                double distance = Math.sqrt(Math.pow(x - centroid[0], 2)
                        + Math.pow(y - centroid[1], 2)
                        + Math.pow(z - centroid[2], 2));

                if (distance < minDist) {
                    minDist = distance;
                    nearestCentroid = i;
                }
            }

            context.write(new IntWritable(nearestCentroid), value);
        }
    }

    public static class SilhouetteReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        private Map<Integer, List<double[]>> allClusters = new HashMap<>();

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            List<double[]> clusterPoints = new ArrayList<>();
            for (Text value : values) {
                String[] point = value.toString().split(",");
                double x = Double.parseDouble(point[0]);
                double y = Double.parseDouble(point[1]);
                double z = Double.parseDouble(point[2]);
                clusterPoints.add(new double[]{x, y, z});
            }

            // Store the points of this cluster
            allClusters.put(key.get(), clusterPoints);
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Calculate silhouette score for all clusters
            for (Map.Entry<Integer, List<double[]>> entry : allClusters.entrySet()) {
                int clusterId = entry.getKey();
                List<double[]> clusterPoints = entry.getValue();
                int count = clusterPoints.size();

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

                double averageIntraDistance = totalIntraDistance / (count * (count - 1));

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
                double silhouetteScore = (averageInterDistance - averageIntraDistance) /
                        Math.max(averageIntraDistance, averageInterDistance);

                context.write(new IntWritable(clusterId), new Text("Avg Intra: " + averageIntraDistance
                        + ", Avg Inter: " + averageInterDistance + ", Silhouette Score: " + silhouetteScore));
            }
        }
    }

    public static void runSilhouetteEvaluation(Configuration conf, String inputPath, String seedFilePath, String outputPath) throws Exception {
        conf.set("seedFilePath", seedFilePath);

        Job job = Job.getInstance(conf, "Silhouette Evaluation");
        job.setJarByClass(SilhouetteEvaluation.class);
        job.setMapperClass(SilhouetteMapper.class);
        job.setReducerClass(SilhouetteReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public static void main(String[] args) throws Exception {

        //Pass the correct file path in this format - "file:///C:/path/to/file/filename"

        String inputPath = ".../Project2/3d_points_dataset.csv";
        String seedFilePath = ".../Project2/seed_points_K5.csv";
        String outputPath = ".../Project2/output/Silhouette1";

        Configuration conf = new Configuration();
        runSilhouetteEvaluation(conf, inputPath, seedFilePath, outputPath);
    }
}
