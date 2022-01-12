using Tensorflow;

namespace TensorFlowDotNetWrapper.DenseLayerConfigBundle
{
    public interface IDenseLayerConfig
    {
        int GetNeuronCount();
        string GetActivationName();
        float? GetDropoutRate();
        float[] GetKernels();
        float[] GetBias();
        Tensors ToTensors(Tensors previousLayer);
    }
}
