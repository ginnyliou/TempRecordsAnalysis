using TensorFlowDotNetWrapper.DenseLayerConfigBundle;
using TensorFlowDotNetWrapper.InputLayerConfigBundle;
using TensorFlowDotNetWrapper.OutputLayerConfigBundle;

namespace TensorFlowDotNetWrapper.DenseModelWrapperBundle
{
    // TODO:
    // - Pass data and store accuracy in double instead.
    public interface IDenseModelWrapper
    {
        IInputLayerConfig GetInputLayerConfig();
        IOutputLayerConfig GetOutputLayerConfig();
        IDenseLayerConfig[] GetHiddenLayerConfigs();
        float? GetAccuracy();
        IDenseModelWrapper Train(float[,] data, float[,] result, int epochs = 1, float validationSplit = 0.25f);
        float[] Predict(float[] data);
    }
}
