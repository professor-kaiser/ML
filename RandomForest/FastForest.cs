using ML.RandomForest.Algorithm;
using ML.RandomForest.Data;
using ML.RandomForest.Structural;

namespace ML.RandomForest;

public class FastForest
{
    public List<DecisionNode> Trees { get; set; } = new();
    private Random Rnd { get; set; } = new();
    public List<ISample>? Samples { get; set; }
    public int TreeCount { get; set; }
    public int[]? Features { get; set; }
    public (int Current, int Max) Depth { get; set; }
    public Func<ISample, int, double> Selector { get; set; } = null!;


    public void Build()
    {
        for (int i = 0; i < TreeCount; i++)
        {
            var boot = Metrics.Bootstrap(Samples!, Rnd);
            Trees.Add(DecisionNode.Build(
                data: (Samples!, Features!),
                depth: Depth,
                selector: Selector,
                Rnd
            ));
        }
    }

    public string Predict(
    ISample sample,
    Func<ISample, int, double> featureAccessor)
    {
        return Trees.Select(t => t.Predict(sample, featureAccessor))
                    .GroupBy(x => x)
                    .OrderByDescending(g => g.Count())
                    .First().Key;
    }


    // public string Predict(ISample sample, Func<ISample, double> selector)
    // {
    //     return Trees.Select(t => t.Predict(sample, selector))
    //                 .GroupBy(x => x)
    //                 .OrderByDescending(g => g.Count())
    //                 .First().Key;
    // }
}