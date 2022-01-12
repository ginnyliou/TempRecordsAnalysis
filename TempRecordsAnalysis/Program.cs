using System;
using System.Linq;
using TensorFlowDotNetWrapper.DenseLayerConfigBundle;
using TensorFlowDotNetWrapper.DenseModelWrapperBundle;
using TensorFlowDotNetWrapper.InputLayerConfigBundle;
using TensorFlowDotNetWrapper.OutputLayerConfigBundle;

namespace TempRecordsAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello tester!");
            var localStorage = new RecordStorage { StorageBasePath = @"D:\TempRecordsAnalysis\records\" }; // Change to match the path.
            var allRecords = localStorage.LoadRecords().OrderByDescending(r => r.RecTime).ToArray();
            var lookbacks = allRecords.AsParallel().Select((r, i) =>
                    new TempRecordLookback()
                    {
                        TargetRecord = r,
                        PreviousRecords = allRecords.Skip(i + 1).Take(10).ToList()
                    }
                )
                .OrderByDescending(l => l.TargetRecord.RecTime)
                .ToArray();
            var inputOutputs = lookbacks.Select(l => l.ToInputsOutputs()).ToArray();
            var inputCount = inputOutputs.FirstOrDefault().Inputs.Length;
            var outputCount = inputOutputs.FirstOrDefault().Outputs.Length;
            if (inputCount <= 0 || outputCount <= 0) return;
            inputOutputs = inputOutputs.Where(io => io.Inputs.Length == inputCount && io.Outputs.Length == outputCount).ToArray();
            var modelWrapper = new DenseModelWrapper(
                new InputLayerConfig(inputCount),
                new OutputLayerConfig(outputCount, OutputLayerConfig.KnownActivation.Softmax.ToString()),
                new[]
                {
                    new DenseLayerConfig(60, DenseLayerConfig.KnownActivation.ReLU.ToString())
                }
            );
            float[,] inputs = new float[inputOutputs.Length, inputCount];
            float[,] outputs = new float[inputOutputs.Length, outputCount];
            for (int i = 0; i < inputOutputs.Length; i++)
            {
                for (int j = 0; j < inputOutputs[i].Inputs.Length; j++) inputs[i, j] = inputOutputs[i].Inputs[j];
                for (int k = 0; k < inputOutputs[i].Outputs.Length; k++) outputs[i, k] = inputOutputs[i].Outputs[k];
            }
            IDenseModelWrapper newModel = modelWrapper.Train(inputs, outputs);
            Console.WriteLine("Bye tester!");
        }
    }
}