using Tensorflow;

namespace TensorFlowDotNetWrapper.InputLayerConfigBundle
{
    public interface IInputLayerConfig
    {
        int GetInputCount();
        Tensors ToTensors();
    }
}
