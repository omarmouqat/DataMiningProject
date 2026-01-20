public class EvaluationResult {
    public int tp, fp, tn, fn;

    public double accuracy;
    public double precision;
    public double recall;
    public double f1;

    @Override
    public String toString() {
        return String.format(
                "Accuracy: %.4f%nPrecision: %.4f%nRecall: %.4f%nF1-score: %.4f%nTP=%d FP=%d TN=%d FN=%d",
                accuracy, precision, recall, f1, tp, fp, tn, fn
        );
    }
}

