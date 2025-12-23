using ML.RandomForest.Data;
using ML.RandomForest.Algorithm;
using System.Linq;

namespace ML.RandomForest.Structural;

public class DecisionNode
{
    public int FeatureIndex;
    public double Threshold;
    public DecisionNode? Left;
    public DecisionNode? Right;
    public string? Label;

    public string Predict(ISample sample, Func<ISample, int, double> selector)
    {
        if (Label != null) return Label;

        if (selector(sample, FeatureIndex) < Threshold)
        {
            return Left!.Predict(sample, selector);
        }
        else
        {
            return Right!.Predict(sample, selector);
        }
    }

    public static DecisionNode Build(
        (List<ISample> Samples, int[] Features) data, 
        (int Current, int Max) depth, 
        Func<ISample, int, double> selector, 
        Random rnd)
    {
        if (depth.Current >= depth.Max || data.Samples.Select(s => s.Label).Distinct().Count() == 1)
        {
            return new DecisionNode
            {
                Label = data.Samples.GroupBy(s => s.Label)
                            .OrderByDescending(g => g.Count())
                            .First().Key
            };
        }

        double bestGain = 0;
        double bestThreshold = 0;
        int bestFeature = -1;
        List<ISample>? bestLeft = null, bestRight = null;
        var parentGini = Metrics.Gini(data.Samples);

        int m = (int)Math.Round(Math.Sqrt(data.Features.Length)); // m = sqrt(F)
        var features = data.Features.OrderBy(x => rnd.Next()).Take(m);

        foreach (var feature in features)
        {
            var values = data.Samples.Select(s => selector(s, feature)).Distinct();
            
            foreach(double threshold in values)
            {
                var left = data.Samples.Where(s => selector(s, feature) < threshold).ToList();
                var right = data.Samples.Where(s => selector(s, feature) >= threshold).ToList();

                if (left.Count == 0 || right.Count == 0) continue;

                double gain = parentGini 
                    - ((double)left.Count / data.Samples.Count) * Metrics.Gini(left)
                    - ((double)right.Count / data.Samples.Count) * Metrics.Gini(right);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestLeft = left;
                    bestRight = right;
                }
            }
        }

        if (bestGain == 0)
        {
            return new DecisionNode 
            { 
                Label = data.Samples[0].Label 
            };
        }

        return new DecisionNode
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            Left = Build(
                data: (bestLeft!, data.Features), 
                depth: (depth.Current + 1, depth.Max), 
                selector: selector,
                rnd
            ),
            Right = Build(
                data: (bestRight!, data.Features), 
                depth: (depth.Current + 1, depth.Max), 
                selector: selector,
                rnd
            ),
        };
    }
}