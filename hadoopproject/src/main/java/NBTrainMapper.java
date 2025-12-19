import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class NBTrainMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // Input Example: p,x,s,n,t,p,f,c... (p = poisonous)
        String line = value.toString();
        String[] attributes = line.split(",");

        // 1. Get the Class Label (e.g., "p" or "e")
        // Assuming the class label is the FIRST column (index 0)
        String classLabel = attributes[0];

        // Emit count for the class itself (Prior Probability)
        // Key: "Class_p", Value: 1
        word.set("Class_" + classLabel);
        context.write(word, one);

        // 2. Iterate through all other features to count Conditional Probabilities
        for (int i = 1; i < attributes.length; i++) {
            // Emit count for Feature + Class
            // Key: "p_Col1_x", Value: 1  (Class p, Column 1 has value x)
            word.set(classLabel + "_Col" + i + "_" + attributes[i]);
            context.write(word, one);
        }
    }
}