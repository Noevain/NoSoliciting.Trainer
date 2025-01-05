using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using ConsoleTables;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using NoSoliciting.Interface;

namespace NoSoliciting.Trainer;

internal static class Program {
    private static readonly string[] StopWords =
    [
        "discord",
        "gg",
        "twitch",
        "tv",
        "lgbt",
        "lgbtq",
        "lgbtqia",
        "http",
        "https",
        "18",
        "come",
        "join",
        "blu",
        "mounts",
        "ffxiv"
    ];

    private enum Mode {
        Test,
        CreateModel,
        Interactive,
        InteractiveFull,
        Normalise,
    }

    [Serializable]
    [JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    private class ReportInput {
        public uint ReportVersion { get; } = 2;
        public uint ModelVersion { get; set; }
        public DateTime Timestamp { get; set; }
        public ushort Type { get; set; }
        public List<byte> Sender { get; set; }
        public List<byte> Content { get; set; }
        public string? Reason { get; set; }
        public string? SuggestedClassification { get; set; }
    }

    private static void Main(string[] args) {
        var mode = args[0] switch {
            "test" => Mode.Test,
            "create-model" => Mode.CreateModel,
            "interactive" => Mode.Interactive,
            "interactive-full" => Mode.InteractiveFull,
            "normalise" => Mode.Normalise,
            _ => throw new ArgumentException("invalid argument"),
        };

        if (mode == Mode.Normalise) {
            Console.WriteLine("Ready");
            while (true) {
                Console.Write("> ");
                var input = Console.ReadLine();
                var bytes = Convert.FromBase64String(input!);
                var toNormalise = Encoding.UTF8.GetString(bytes);
                var normalised = NoSolUtil.Normalise(toNormalise);
                Console.WriteLine(normalised);
            }
        }

        var path = "../../../data.csv";
        if (args.Length > 1) {
            path = args[1];
        }

        var parentDir = Directory.GetParent(path);
        if (parentDir == null) {
            throw new ArgumentException("data.csv did not have a parent directory");
        }

        var ctx = new MLContext(seed: 1);

        // =========================
        // Load CSV into Data
        // =========================
        List<Data> records;
        using (var fileStream = new FileStream(path, FileMode.Open)) {
            using var stream = new StreamReader(fileStream);
            using var csv = new CsvReader(stream, new CsvConfiguration(CultureInfo.InvariantCulture) {
                HeaderValidated = null,
            });

            records = csv
                .GetRecords<Data>()
                .Select(rec => {
                    // Clean up some weird characters
                    rec.Message = rec.Message
                        .Replace("", "")
                        .Replace("", "")
                        .Replace("\r\n", " ")
                        .Replace("\r", " ")
                        .Replace("\n", " ");
                    return rec;
                })
                .OrderBy(rec => rec.Category)
                .ThenBy(rec => rec.Channel)
                .ThenBy(rec => rec.Message)
                .ToList();
        }

        // Re-save (original code)
        using (var fileStream = new FileStream(path, FileMode.Create)) {
            using var stream = new StreamWriter(fileStream);
            using var csv = new CsvWriter(stream, new CsvConfiguration(CultureInfo.InvariantCulture) {
                NewLine = "\n",
            });
            csv.WriteRecords(records);
        }

        // =========================
        // Class Weights
        // =========================
        var classes = new Dictionary<string, uint>();
        foreach (var record in records) {
            if (!classes.ContainsKey(record.Category!)) {
                classes[record.Category!] = 0;
            }
            classes[record.Category!] += 1;
        }

        var weights = new Dictionary<string, float>();
        foreach (var (category, count) in classes) {
            var nSamples = (float) records.Count;
            var nClasses = (float) classes.Count;
            var w = nSamples / (nClasses * count);

            // Heavily boost NORMAL
            if (category == "NORMAL") {
                w *= 8.0f;
            }
            weights[category] = w;
        }

        // =========================
        // STAGE 1: BINARY
        // =========================
        var binaryRecords = records.Select(r => new DataBinary {
            Channel = (ushort)r.Channel,
            Message = r.Message,
            IsNormal = (r.Category == "NORMAL")
        }).ToList();

        var dfBinary = ctx.Data.LoadFromEnumerable(binaryRecords);
        var splitBinary = ctx.Data.TrainTestSplit(dfBinary, 0.2, seed: 1);

        var pipelineBinary =
            ctx.Transforms.Text.NormalizeText("MsgNormal", nameof(DataBinary.Message), 
                    keepPunctuations: false, keepNumbers: false)
                .Append(ctx.Transforms.Text.TokenizeIntoWords("MsgTokens", "MsgNormal"))
                .Append(ctx.Transforms.Text.RemoveDefaultStopWords("MsgNoDefStop", "MsgTokens"))
                .Append(ctx.Transforms.Text.RemoveStopWords("MsgNoStop", "MsgNoDefStop", StopWords))
                .Append(ctx.Transforms.Conversion.MapValueToKey("MsgKey", "MsgNoStop"))
                .Append(ctx.Transforms.Text.ProduceNgrams("MsgNgrams", "MsgKey",
                    weighting: NgramExtractingEstimator.WeightingCriteria.Tf))
                .Append(ctx.Transforms.NormalizeLpNorm("Features", "MsgNgrams"))
                .Append(ctx.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: nameof(DataBinary.IsNormal),
                    featureColumnName: "Features"));

        var trainBinary = (mode == Mode.CreateModel) ? dfBinary : splitBinary.TrainSet;
        var modelBinary = pipelineBinary.Fit(trainBinary);

        // Evaluate Stage 1 in test mode
        if (mode == Mode.Test) {
            var predsBinary = modelBinary.Transform(splitBinary.TestSet);
            var evalBinary = ctx.BinaryClassification.Evaluate(
                predsBinary, labelColumnName: nameof(DataBinary.IsNormal));

            Console.WriteLine("=== STAGE 1: Normal vs. Not Normal ===");
            Console.WriteLine($" Accuracy: {evalBinary.Accuracy:P3}");
            Console.WriteLine($" AUC: {evalBinary.AreaUnderRocCurve:P3}");
            Console.WriteLine($" F1Score: {evalBinary.F1Score:P3}");
            Console.WriteLine();
            
            // =============== ADDED LOGGING FOR MISCLASSIFICATIONS ===============
            int falsePositives = 0, falseNegatives = 0, truePositives = 0, trueNegatives = 0;
            var dataView = ctx.Data.CreateEnumerable<DataBinary>(splitBinary.TestSet, reuseRowObject: false);
            var predictionEngine = ctx.Model.CreatePredictionEngine<DataBinary, PredictionBinary>(modelBinary);

            foreach (var data in dataView) {
                var prediction = predictionEngine.Predict(data);

                if (data.IsNormal) {
                    if (prediction.PredictedIsNormal) {
                        truePositives++; // Correctly identified as NORMAL
                    } else {
                        falseNegatives++; // NON NORMAL predicted as NORMAL
                    }
                } else {
                    if (prediction.PredictedIsNormal) {
                        falsePositives++; // NORMAL predicted as NON NORMAL
                    } else {
                        trueNegatives++; // Correctly identified as NON NORMAL
                    }
                }
            }

            Console.WriteLine("=== STAGE 1 MISCLASSIFICATIONS ===");
            Console.WriteLine($" True Positives (NORMAL -> NORMAL): {truePositives}");
            Console.WriteLine($" True Negatives (NOT NORMAL -> NOT NORMAL): {trueNegatives}");
            Console.WriteLine($" False Positives (NORMAL -> NOT NORMAL): {falsePositives}");
            Console.WriteLine($" False Negatives (NOT NORMAL -> NORMAL): {falseNegatives}");
            Console.WriteLine($" False Positive Rate (FPR): {(float)falsePositives / (falsePositives + trueNegatives):P3}");
            Console.WriteLine($" False Negative Rate (FNR): {(float)falseNegatives / (falseNegatives + truePositives):P3}");
            Console.WriteLine();
            // ===================================================================
        }

        // =========================
        // STAGE 2: MULTICLASS FOR NOT-NORMAL
        // =========================
        var notNormalRecords = records.Where(r => r.Category != "NORMAL").ToList();
        var dfMulti = ctx.Data.LoadFromEnumerable(notNormalRecords);
        var splitMulti = ctx.Data.TrainTestSplit(dfMulti, 0.2, seed: 1);

        // Reuse your existing transforms
        ctx.ComponentCatalog.RegisterAssembly(typeof(Data).Assembly);
        var compute = new Data.ComputeContext(weights);
        var normalise = new Data.Normalise();

        var pipelineMulti = ctx.Transforms.Conversion.MapValueToKey("Label", nameof(Data.Category))
            .Append(ctx.Transforms.CustomMapping(compute.GetMapping(), "Compute"))
            .Append(ctx.Transforms.CustomMapping(normalise.GetMapping(), "Normalise"))
            .Append(ctx.Transforms.Text.NormalizeText("MsgNormal",
                nameof(Data.Normalise.Normalised.NormalisedMessage),
                keepPunctuations: false,
                keepNumbers: false))
            .Append(ctx.Transforms.Text.TokenizeIntoWords("MsgTokens", "MsgNormal"))
            .Append(ctx.Transforms.Text.RemoveDefaultStopWords("MsgNoDefStop", "MsgTokens"))
            .Append(ctx.Transforms.Text.RemoveStopWords("MsgNoStop", "MsgNoDefStop", StopWords))
            .Append(ctx.Transforms.Conversion.MapValueToKey("MsgKey", "MsgNoStop"))
            .Append(ctx.Transforms.Text.ProduceNgrams("MsgNgrams", "MsgKey",
                weighting: NgramExtractingEstimator.WeightingCriteria.Tf))
            .Append(ctx.Transforms.NormalizeLpNorm("FeaturisedMessage", "MsgNgrams"))
            .Append(ctx.Transforms.Conversion.ConvertType("CPartyFinder", nameof(Data.Computed.PartyFinder)))
            .Append(ctx.Transforms.Conversion.ConvertType("CShout", nameof(Data.Computed.Shout)))
            .Append(ctx.Transforms.Conversion.ConvertType("CTrade", nameof(Data.Computed.ContainsTradeWords)))
            .Append(ctx.Transforms.Conversion.ConvertType("CSketch", nameof(Data.Computed.ContainsSketchUrl)))
            .Append(ctx.Transforms.Conversion.ConvertType("HasWard", nameof(Data.Computed.ContainsWard)))
            .Append(ctx.Transforms.Conversion.ConvertType("HasPlot", nameof(Data.Computed.ContainsPlot)))
            .Append(ctx.Transforms.Conversion.ConvertType("HasNumbers", nameof(Data.Computed.ContainsHousingNumbers)))
            .Append(ctx.Transforms.Concatenate("Features", 
                "FeaturisedMessage", "CPartyFinder", "CShout", "CTrade",
                "HasWard", "HasPlot", "HasNumbers", "CSketch"))
            .Append(ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "Label",
                featureColumnName: "Features",
                exampleWeightColumnName: "Weight"
            ))
            .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var trainMulti = (mode == Mode.CreateModel) ? dfMulti : splitMulti.TrainSet;
        var modelMulti = pipelineMulti.Fit(trainMulti);

        // Save both sub-models into one .zip
        if (mode == Mode.CreateModel) {
            var singleZipPath = Path.Join(parentDir.FullName, "model.zip");

            using (var file = File.Create(singleZipPath))
            using (var archive =
                   new System.IO.Compression.ZipArchive(file, System.IO.Compression.ZipArchiveMode.Create))
            {
                var binaryEntry = archive.CreateEntry("model_binary.zip");
                using (var stream = binaryEntry.Open()) {
                    ctx.Model.Save(modelBinary, trainBinary.Schema, stream);
                }

                var multiEntry = archive.CreateEntry("model_multiclass.zip");
                using (var stream = multiEntry.Open()) {
                    ctx.Model.Save(modelMulti, trainMulti.Schema, stream);
                }
            }
        }

        // Evaluate Stage 2 in test mode
        if (mode == Mode.Test) {
            var predsMulti = modelMulti.Transform(splitMulti.TestSet);
            var evalMulti = ctx.MulticlassClassification.Evaluate(predsMulti);

            Console.WriteLine("=== STAGE 2: (Not Normal) Multiclass ===");
            Console.WriteLine($" Macro acc: {evalMulti.MacroAccuracy * 100:F3}");
            Console.WriteLine($" Micro acc: {evalMulti.MicroAccuracy * 100:F3}");
            Console.WriteLine($" Log loss : {evalMulti.LogLoss * 100:F3}");
            Console.WriteLine();

            // =============== ADDED TABLE CODE (Detailed Confusion Matrix) ===============
            // This prints a table with rows=expected, columns=predicted, just for stage-2 classes
            var slotNames2 = new VBuffer<ReadOnlyMemory<char>>();
            // We'll create a temporary prediction engine just for slot names, or
            // we can grab from predsMulti.Schema if needed
            predsMulti.Schema["Score"].GetSlotNames(ref slotNames2);
            var names2 = slotNames2.DenseValues().Select(s => s.ToString()).ToList();

            var cols2 = new string[1 + names2.Count];
            cols2[0] = "";
            for (var i = 0; i < names2.Count; i++) {
                cols2[i+1] = names2[i];
            }

            var table2 = new ConsoleTable(cols2);

            // The confusion matrix is 2D: evalMulti.ConfusionMatrix.Counts[row][col]
            for (var i = 0; i < names2.Count; i++) {
                var name = names2[i];
                var confuseRow = evalMulti.ConfusionMatrix.Counts[i];
                var row = new object[1 + confuseRow.Count];
                row[0] = name;
                for (int j = 0; j < confuseRow.Count; j++) {
                    if (i == j) {
                        row[j+1] = $"= {confuseRow[j]} =";
                    } else {
                        row[j+1] = confuseRow[j];
                    }
                }
                table2.AddRow(row);
            }

            Console.WriteLine("Rows = expected class, columns = predicted class");
            Console.WriteLine(table2.ToString());
            // =============== END ADDED TABLE CODE ===============
        }

        // =========================
        // If we're only test/create-model, exit
        // =========================
        switch (mode) {
            case Mode.Test:
            case Mode.CreateModel:
                return;
        }

        // =========================
        // INTERACTIVE
        // =========================

        var predEngineBinary = ctx.Model.CreatePredictionEngine<DataBinary, PredictionBinary>(modelBinary);
        var predEngineMulti  = ctx.Model.CreatePredictionEngine<Data, Prediction>(modelMulti);

        // We can read the class names for debug if desired
        var slotNames = new VBuffer<ReadOnlyMemory<char>>();
        predEngineMulti.OutputSchema["Score"].GetSlotNames(ref slotNames);
        var multiClassNames = slotNames.DenseValues().Select(x => x.ToString()).ToList();

        Console.WriteLine("Interactive mode: Enter `<channel> <Base64Message>`.  Empty line to exit.");

        while (true) {
            var line = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) break;

            var parts = line.Split(' ', 2);
            if (parts.Length < 2 || !ushort.TryParse(parts[0], out var channel)) {
                continue;
            }

            var base64Msg = parts[1];
            var size = Base64.GetMaxDecodedFromUtf8Length(base64Msg.Length);
            var buf = new byte[size];
            if (Convert.TryFromBase64String(base64Msg, buf, out var written)) {
                base64Msg = Encoding.UTF8.GetString(buf.AsSpan(0, written));
            }

            // Stage 1
            var binInput = new DataBinary {
                Channel = channel,
                Message = base64Msg
            };
            var binPred = predEngineBinary.Predict(binInput);

            // =============== ADDED THRESHOLD OVERRIDE ===============
            // "If in doubt, classify as NORMAL."
            // We can override the default (true/false) by checking Probability(IsNormal).
            // e.g. if Probability >= 0.4 => normal
            // Tweak the threshold to tilt borderline spam → normal.

            float threshold = 0.3f;
            bool predictedIsNormal;
            if (binPred.Probability >= threshold) {
                predictedIsNormal = true;
            } else {
                predictedIsNormal = false;
            }
            // =========================================================

            if (predictedIsNormal) {
                // "Borderline spam" or "Definite normal" => Normal
                Console.WriteLine("NORMAL");
            }
            else {
                // If not normal, Stage 2
                var dataInput = new Data(channel, base64Msg);
                var multiPred = predEngineMulti.Predict(dataInput);

                Console.WriteLine(multiPred.Category);

                // optional: see the probabilities
                for (int i = 0; i < multiClassNames.Count; i++) {
                    Console.WriteLine($"  {multiClassNames[i]}: {multiPred.Probabilities[i] * 100:F2}%");
                }
            }
        }
    }
}