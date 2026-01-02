import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Map;

public class NBMapReduce {

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

    public String predict(String line, Map<String,Integer> model){
        String[] attributes = line.split(",");
        int NbrClassp = model.get("Class_p");
        int NbrClasse = model.get("Class_e");

        int total = NbrClassp + NbrClasse ;
        double probaP = Math.log(NbrClassp / (double)total) ;
        double probaE = Math.log(NbrClasse / (double)total) ;

        System.out.println(NbrClasse);
        System.out.println(NbrClassp);
        System.out.println(probaP);
        System.out.println(probaE);
        for(int i = 0; i < attributes.length; i++){
                String Pkey  = "p_Col" + (i + 1) + "_" + attributes[i];
                String Ekey  = "e_Col" + (i + 1) + "_" + attributes[i];
                if(model.containsKey(Pkey)){
                    probaP += Math.log(model.get(Pkey) / (double) NbrClassp);
                }else{
                    probaP += 0.000000000001;
                }
                if(model.containsKey(Ekey)){
                    probaE += Math.log(model.get(Ekey) / (double) NbrClasse);
                }else{
                    probaE += 0.000000000001;
                }
        }
        System.out.println("ProbP:"+probaP);
        System.out.println("ProbE:"+probaE);
        if(probaP > probaE) return "p";
        return "e";
    }

}
