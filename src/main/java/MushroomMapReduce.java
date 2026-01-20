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

public class MushroomMapReduce {
    public static class NBModel {
        public Map<String, Integer> featureCounts = new HashMap<>();
        public Map<String, Integer> classCounts = new HashMap<>();
        public Map<String, Integer> classFeatureTotals = new HashMap<>();
        public Set<String> featureVocabulary = new HashSet<>();
    }
    public static class NBMapper
            extends Mapper<LongWritable, Text, Text, IntWritable> {

        private final Text outKey = new Text();
        private static final IntWritable ONE = new IntWritable(1);

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] attributes = value.toString().split(",");

            String classLabel = attributes[0];
            outKey.set("Class_" + classLabel);
            context.write(outKey, ONE);

            for (int i = 1; i < attributes.length; i++) {
                outKey.set(classLabel + "_Col" + i + "_" + attributes[i]);
                context.write(outKey, ONE);
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

    public static String predict(String[] features, NBModel model) {
        String bestClass = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        // Total docs for prior
        int totalDocs = model.classCounts.values().stream().mapToInt(i -> i).sum();

        for (String clazz : model.classCounts.keySet()) {
            double logProb = Math.log((double) model.classCounts.get(clazz) / totalDocs);

            for (int i = 0; i < features.length; i++) {
                String featureKey = clazz + "_Col" + (i + 1) + "_" + features[i];
                int count = model.featureCounts.getOrDefault(featureKey, 0);
                int totalFeatureCount = model.classFeatureTotals.getOrDefault(clazz, 0);
                int vocabSize = model.featureVocabulary.size();

                double prob = (count + 1.0) / (totalFeatureCount + vocabSize);
                logProb += Math.log(prob);
            }

            if (logProb > bestScore) {
                bestScore = logProb;
                bestClass = clazz;
            }
        }

        return bestClass;
    }

    public static NBModel loadModel(String modelPath, Configuration conf) throws IOException {
        NBModel model = new NBModel();
        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(modelPath);

        FileStatus[] files = fs.listStatus(path, p -> p.getName().startsWith("part-"));

        for (FileStatus file : files) {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(fs.open(file.getPath())))) {

                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if(line.isEmpty()) continue;
                    String[] parts = line.split("\\s+");
                    if(parts.length != 2) {
                        System.err.println("Skipping malformed line: " + line);
                        continue;
                    }

                    String key = parts[0];
                    int count;
                    try {
                        count = Integer.parseInt(parts[1]);
                    } catch(NumberFormatException e) {
                        System.err.println("Skipping line with invalid count: " + line);
                        continue;
                    }

                    if(key.startsWith("Class_")) {
                        String clazz = key.substring(6);
                        model.classCounts.put(clazz, count);
                    } else {
                        String[] keyParts = key.split("_Col");
                        if(keyParts.length < 2) {
                            System.err.println("Skipping invalid feature key: " + key);
                            continue;
                        }
                        String clazz = keyParts[0];
                        String rest = keyParts[1]; // e.g., "10_e"
                        String featureKey = clazz + "_Col" + rest;
                        model.featureCounts.put(featureKey, count);
                        model.featureVocabulary.add(featureKey);
                        model.classFeatureTotals.merge(clazz, count, Integer::sum);
                    }

                }
            }
        }
        return model;
    }


    public static EvaluationResult evaluateModelFromHDFS(
            String hdfsTestPath, NBModel model, Configuration conf) throws IOException {

        EvaluationResult result = new EvaluationResult();
        FileSystem fs = FileSystem.get(conf);
        Path path = new Path(hdfsTestPath);
        List<Path> files = new ArrayList<>();

        if (fs.isDirectory(path)) {
            for (FileStatus status : fs.listStatus(path)) {
                if (status.getPath().getName().startsWith("part-"))
                    files.add(status.getPath());
            }
        } else files.add(path);

        for (Path p : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(p)))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] attributes = line.split(",");
                    if (attributes.length < 2) continue;

                    String trueClass = attributes[0];
                    String[] features = Arrays.copyOfRange(attributes, 1, attributes.length);
                    String predicted = predict(features, model);

                    if (predicted.equals(trueClass)) {
                        result.tp++;
                    } else {
                        result.fp++;
                        result.fn++;
                    }
                }
            }
        }

        int total = result.tp + result.fp;
        result.accuracy = (double) result.tp / total;
        result.precision = result.tp == 0 ? 0 : (double) result.tp / (result.tp + result.fp);
        result.recall = result.tp == 0 ? 0 : (double) result.tp / (result.tp + result.fn);
        result.f1 = (result.precision + result.recall) == 0 ? 0 :
                2 * result.precision * result.recall / (result.precision + result.recall);

        return result;
    }

}
