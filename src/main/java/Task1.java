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

public class Task1 {

    public static class CentroidMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private List<double[]> centroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // Load centroids from seed_points_K5.csv
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

                System.out.println("Centroid " + i + ": (" + centroid[0] + "," + centroid[1] + "," + centroid[2] + "), Distance: " + distance);

                if (distance < minDist) {
                    minDist = distance;
                    nearestIndex = i;
                }
            }

            System.out.println("Nearest centroid for point (" + x + "," + y + "," + z + ") is: " + nearestIndex);
            return nearestIndex;
        }


        private List<double[]> loadCentroids(String seedFilePath) {
            List<double[]> centroids = new ArrayList<>();
            try {
                FileSystem fs = FileSystem.get(new Configuration());
                Path path = new Path(seedFilePath);
                BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));

                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split(",");
                    double[] centroid = new double[3];
                    centroid[0] = Double.parseDouble(parts[0]);
                    centroid[1] = Double.parseDouble(parts[1]);
                    centroid[2] = Double.parseDouble(parts[2]);
                    centroids.add(centroid);
                }
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
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

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        //Pass the correct file path in this format - "file:///C:/path/to/file/filename"

        conf.set("seedFilePath", ".../Project2/seed_points_K5.csv"); // Pass seed points file path as argument

        Job job = Job.getInstance(conf, "KMeans Single Iteration");
        job.setJarByClass(Task1.class);
        job.setMapperClass(CentroidMapper.class);
        job.setReducerClass(CentroidReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        //Pass the correct file path in this format - "file:///C:/path/to/file/filename"

        FileInputFormat.addInputPath(job, new Path(".../Project2/3d_points_dataset.csv"));  // Input file
        FileOutputFormat.setOutputPath(job, new Path(".../Project2/output/task1"));  // Output directory

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
