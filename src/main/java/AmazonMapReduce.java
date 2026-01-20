import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class AmazonMapReduce {
    public static class NBModel {
        public Map<String, Integer> wordCounts = new HashMap<>();
        public Map<String, Integer> classDocCounts = new HashMap<>();
        public Map<String, Integer> classWordTotals = new HashMap<>();
        public Set<String> vocabulary = new HashSet<>();
    }
    public static class NBMapper
            extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);
        private Text outKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            // Correct CSV parsing
            String[] parts = value.toString().split(",");
            if (parts.length < 2) return;

            String ratingStr = parts[0];
            String review = parts[1];

            int rating;
            try {
                rating = Integer.parseInt(ratingStr);
            } catch (NumberFormatException e) {
                return;
            }

            // Sentiment labels
            String label = (rating == 2) ? "positive" : "negative";

            // Count documents per class
            outKey.set(label + ",__DOC__");
            context.write(outKey, ONE);

            // Emit word counts
            for (String token : review.split("\\s+")) {

                if (!token.isEmpty()) {
                    outKey.set(label + "," + token);
                    context.write(outKey, ONE);
                }
            }
        }
    }

    public static class NBReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable result = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable v : values) {
                sum += v.get();
            }

            result.set(sum);
            context.write(key, result);
        }
    }

    // Prediction logic is correct and does NOT need MapReduce
    public static String predict(String review, NBModel model) {

        String bestClass = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        int vocabSize = model.vocabulary.size();

        for (String clazz : model.classDocCounts.keySet()) {

            // Prior: log P(class)
            double logProb = Math.log(model.classDocCounts.get(clazz));

            for (String token : review.toLowerCase().split("\\s+")) {
                token = token.replaceAll("[^a-z]", "");
                if (token.isEmpty()) continue;

                String key = clazz + "," + token;
                int wordCount = model.wordCounts.getOrDefault(key, 0);

                // Laplace smoothing
                double prob = (wordCount + 1.0) /
                        (model.classWordTotals.get(clazz) + vocabSize);

                logProb += Math.log(prob);
            }

            if (logProb > bestScore) {
                bestScore = logProb;
                bestClass = clazz;
            }
        }

        return bestClass;
    }

    public static NBModel loadModel(String modelPath, Configuration conf) throws Exception {

        NBModel model = new NBModel();
        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(modelPath);

        // Read all part-* files
        FileStatus[] files = fs.listStatus(path, p -> p.getName().startsWith("part-"));

        for (FileStatus file : files) {
            FSDataInputStream in = fs.open(file.getPath());
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));

            String line;
            while ((line = reader.readLine()) != null) {
                // class,token \t count
                String[] parts = line.split("\t");
                if (parts.length != 2) continue;

                String key = parts[0];
                int count = Integer.parseInt(parts[1]);

                String[] keyParts = key.split(",");

                String clazz = keyParts[0];
                String token = keyParts[1];

                if (token.equals("__DOC__")) {
                    // Document count per class
                    model.classDocCounts.put(clazz, count);
                } else {
                    // Word count per class
                    model.wordCounts.put(clazz + "," + token, count);
                    model.vocabulary.add(token);

                    model.classWordTotals.merge(clazz, count, Integer::sum);
                }
            }
            reader.close();
        }

        return model;
    }


    public static EvaluationResult evaluateModelFromHDFS(
            String hdfsTestPath,
            NBModel model,
            Configuration conf) throws IOException {

        EvaluationResult result = new EvaluationResult();

        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(hdfsTestPath);

        // Handle directory (multiple part files)
        List<Path> files = new ArrayList<>();
        if (fs.isDirectory(path)) {
            for (FileStatus status : fs.listStatus(path)) {
                if (status.getPath().getName().startsWith("part-")) {
                    files.add(status.getPath());
                }
            }
        } else {
            files.add(path);
        }

        for (Path p : files) {
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(fs.open(p)))) {

                String line;
                while ((line = br.readLine()) != null) {

                    // Robust CSV split
                    String[] parts = line.split("\",\"");
                    if (parts.length < 3) continue;

                    String ratingStr = parts[0].replace("\"", "").trim();
                    String review = parts[2].replace("\"", "").trim();

                    int rating;
                    try {
                        rating = Integer.parseInt(ratingStr);
                    } catch (NumberFormatException e) {
                        continue;
                    }

                    if (rating == 3) continue;

                    String trueLabel = (rating == 2) ? "positive" : "negative";
                    String predicted = predict(review, model);

                    if ("positive".equals(predicted)) {
                        if ("positive".equals(trueLabel)) result.tp++;
                        else result.fp++;
                    } else {
                        if ("negative".equals(trueLabel)) result.tn++;
                        else result.fn++;
                    }
                }
            }
        }

        int total = result.tp + result.fp + result.tn + result.fn;

        result.accuracy = (double) (result.tp + result.tn) / total;
        result.precision = result.tp == 0 ? 0.0 :
                (double) result.tp / (result.tp + result.fp);
        result.recall = result.tp == 0 ? 0.0 :
                (double) result.tp / (result.tp + result.fn);
        result.f1 = (result.precision + result.recall) == 0 ? 0.0 :
                2 * result.precision * result.recall /
                        (result.precision + result.recall);

        return result;
    }


}
