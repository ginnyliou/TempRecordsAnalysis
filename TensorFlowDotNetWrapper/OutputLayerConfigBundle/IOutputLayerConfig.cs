using Tensorflow;

namespace TensorFlowDotNetWrapper.OutputLayerConfigBundle
{
    public interface IOutputLayerConfig
    {
        int GetOutputCount();
        string GetActivationName();
        float[] GetKernels();
        float[] GetBias();
        Tensors ToTensors(Tensors previousLayer);
    }
}
