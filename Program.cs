using ML.RandomForest;
using ML.RandomForest.Data;

class Program
{
    static void Main()
    {
        Console.WriteLine("=== Random Forest maison - Test ===\n");

        // 1️⃣ Dataset d'entraînement
        var dataset = new List<ISample>
        {
            // MUSIC
            new MediaSample { Duration = 180,  Size = 5,   HasVideo = 0, Label = "Music" },
            new MediaSample { Duration = 240,  Size = 6,   HasVideo = 0, Label = "Music" },
            new MediaSample { Duration = 210,  Size = 4.5, HasVideo = 0, Label = "Music" },

            // SERIE
            new MediaSample { Duration = 1800, Size = 350, HasVideo = 1, Label = "Serie" },
            new MediaSample { Duration = 2400, Size = 450, HasVideo = 1, Label = "Serie" },
            new MediaSample { Duration = 2700, Size = 500, HasVideo = 1, Label = "Serie" },

            // MOVIE
            new MediaSample { Duration = 5400, Size = 1300, HasVideo = 1, Label = "Movie" },
            new MediaSample { Duration = 6000, Size = 1500, HasVideo = 1, Label = "Movie" },
            new MediaSample { Duration = 7200, Size = 2000, HasVideo = 1, Label = "Movie" },
        };

        // 2️⃣ Entraînement du Random Forest
        Console.WriteLine("Entraînement du Random Forest...\n");
        var forest = new FastForest
        {
            Samples = dataset,
            TreeCount = 10,
            Features = [0, 1, 2],
            Depth = (0, 5),
            Selector = (s, feature) => feature switch
                {
                    0 => ((MediaSample)s).Duration,
                    1 => ((MediaSample)s).Size,
                    2 => ((MediaSample)s).HasVideo,
                    _ => throw new InvalidOperationException()
                }
        };

        forest.Build();

        // 3️⃣ Fichiers à tester
        var tests = new List<MediaSample>
        {
            new() { Duration = 200,  Size = 5,   HasVideo = 0 },
            new() { Duration = 2300, Size = 420, HasVideo = 1 },
            new() { Duration = 6500, Size = 1600,HasVideo = 1 },
            new() { Duration = 900,  Size = 50,  HasVideo = 0 }, // cas limite
        };

        // 4️⃣ Prédictions
        Console.WriteLine("Résultats des prédictions :\n");

        foreach (var sample in tests)
        {
            string prediction = forest.Predict(
                sample,
                (s, feature) => feature switch
                {
                    0 => ((MediaSample)s).Duration,
                    1 => ((MediaSample)s).Size,
                    2 => ((MediaSample)s).HasVideo,
                    _ => throw new InvalidOperationException()
                });

            Console.WriteLine(
                $"Durée={sample.Duration}s | Taille={sample.Size}MB | Video={sample.HasVideo} " +
                $"=> Prédit : {prediction}"
            );
        }

        Console.WriteLine("\n=== Fin du test ===");
    }

    public class MediaSample : ISample
    {
        public double Duration;   // secondes
        public double Size;       // MB
        public int HasVideo;      // 0 ou 1
        public string Label { get; set; }
    }
}