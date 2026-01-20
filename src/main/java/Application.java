import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Duration;
import java.util.*;

public class Application {

    public static void main(String[] args) throws Exception {

    trainAndTestAmazonReview();
    }

    public static void trainAndTestMushroom() throws Exception{
        long NBStart = System.nanoTime();
        runMapReduce( MushroomMapReduce.NBMapper.class, MushroomMapReduce.NBReducer.class,"MushroomNBMapReduce","/dataset/mushroom.data");
        long NBEnd = System.nanoTime();

        System.out.println("NB duration");
        printDuration(NBStart, NBEnd);
        MushroomMapReduce.NBModel model = MushroomMapReduce.loadModel("/output/MushroomNBMapReduce",new Configuration());
        System.out.println("Test:");
        String tests =  MushroomMapReduce.evaluateModelFromHDFS("/dataset/mushroom.data",model,new Configuration()).toString();
        System.out.println(tests);
        while(true){
            Scanner scanner = new Scanner(System.in);
            String line = scanner.nextLine();
            String prediction =  MushroomMapReduce.predict(line.split(","),model);
            System.out.println(prediction);
        }
    }

    public static void trainAndTestAmazonReview() throws Exception{
        long NBStart = System.nanoTime();
//        runMapReduce(AmazonMapReduce.NBMapper.class,AmazonMapReduce.NBReducer.class,"AmazonReviewNBMapReduce","/dataset/proce_train.csv");
        long NBEnd = System.nanoTime();

        System.out.println("NB duration");
        printDuration(NBStart, NBEnd);
        AmazonMapReduce.NBModel model = AmazonMapReduce.loadModel("/output/AmazonReviewNBMapReduce",new Configuration());
        System.out.println("Test:");
        String tests = AmazonMapReduce.evaluateModelFromHDFS("/dataset/test.csv",model,new Configuration()).toString();
        System.out.println(tests);
        while(true){
            Scanner scanner = new Scanner(System.in);
            String line = scanner.nextLine();
            String prediction = AmazonMapReduce.predict(line,model);
            System.out.println(prediction);
        }
    }


    public static void printDuration(long start,long end){
        Duration duration = Duration.ofNanos(end - start);

        long minutes = duration.toMinutes();
        long seconds = duration.minusMinutes(minutes).getSeconds();
        long milliseconds = duration.minusMinutes(minutes)
                .minusSeconds(seconds)
                .toMillis();

        System.out.println("Execution time : "+minutes + "m " + seconds + "s " + milliseconds + "ms");

    }



    public static void runMapReduce(Class< ? extends Mapper> mapperClass, Class<? extends Reducer> reducerClass,String name,String datasetPath) throws Exception{
        Configuration conf = new Configuration();
        conf.set("mapreduce.input.fileinputformat.split.maxsize", "33554432");
        Job job = Job.getInstance(conf, name);

        job.setJarByClass(Application.class);

        job.setMapperClass(mapperClass);
        job.setReducerClass(reducerClass);
        job.setCombinerClass(reducerClass);
        job.setNumReduceTasks(2);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
//        job.getConfiguration().set("mapreduce.output.textoutputformat.separator", ",");
        FileInputFormat.addInputPath(job, new Path(datasetPath));
        FileOutputFormat.setOutputPath(job, new Path("/output/"+name));
        job.waitForCompletion(true);
    }
    public static Map<String,Integer> getModel(String ModelPath) throws Exception{
        Configuration conf = new Configuration();

        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(ModelPath);

        Map<String,Integer> model = new HashMap<>();
        try (FSDataInputStream inputStream = fs.open(path);
             BufferedReader reader =
                     new BufferedReader(new InputStreamReader(inputStream))) {

            String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                model.put(fields[0], Integer.parseInt(fields[1]));
            }
        }

        fs.close();
        return model;
    }


}
