using ML.RandomForest.Data;

namespace ML.RandomForest.Algorithm;

public static class Metrics
{
    /// <summary>
    ///     gini = 1 - ∑ p_i² | i -> 1...k
    /// </summary>
    /// <param name="samples"></param>
    /// <returns></returns>
    public static double Gini(List<ISample> samples)
    {
        int n = samples.Count;
        var groups = samples.GroupBy(s => s.Label);

        double gini = 1.0;
        foreach (var g in groups)
        {
            double p = (double)g.Count() / n;
            gini -= p*p;
        }

        return gini;
    }

    public static List<ISample> Bootstrap(List<ISample> data, Random rnd)
    {
        var sample = new List<ISample>();

        for (int i = 0; i < data.Count; i++)
        {
            sample.Add(data[rnd.Next(data.Count)]);
        }

        return sample;
    }
}