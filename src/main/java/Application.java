import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Application {

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
//        Job job = Job.getInstance(conf, "NBMapReduce");
//
//        job.setJarByClass(Application.class);
//
//        job.setMapperClass(NBMapReduce.NBMapper.class);
//        job.setReducerClass(NBMapReduce.NBReducer.class);
//        job.setCombinerClass(NBMapReduce.NBReducer.class);
//
//        job.setMapOutputKeyClass(Text.class);
//        job.setMapOutputValueClass(IntWritable.class);
//        job.setOutputKeyClass(Text.class);
//        job.setOutputValueClass(IntWritable.class);
//        job.getConfiguration().set("mapreduce.output.textoutputformat.separator", ",");
//        FileInputFormat.addInputPath(job, new Path("/dataset/mushroom.data"));
//        FileOutputFormat.setOutputPath(job, new Path("/output"));
//        job.waitForCompletion(true);

        FileSystem fs = FileSystem.get(conf);
        Path path = new Path("/output/part-r-00000");

        Map<String,Integer> model = new HashMap<>();
        try (FSDataInputStream inputStream = fs.open(path);
             BufferedReader reader =
                     new BufferedReader(new InputStreamReader(inputStream))) {

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
                String[] fields = line.split(",");
                model.put(fields[0], Integer.parseInt(fields[1]));
            }
        }

        fs.close();

        NBMapReduce mapReduce = new NBMapReduce();
        while(true){
            Scanner scanner = new Scanner(System.in);
            String line = scanner.nextLine();
            String prediction = mapReduce.predict(line,model);
            System.out.println(prediction);
        }


    }
}
