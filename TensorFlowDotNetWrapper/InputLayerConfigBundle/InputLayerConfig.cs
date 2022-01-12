using System;
using Tensorflow;
using Tensorflow.Keras;

namespace TensorFlowDotNetWrapper.InputLayerConfigBundle
{
    public class InputLayerConfig : IInputLayerConfig
    {
        private readonly int _inputCount;

        public InputLayerConfig(int inputCount)
        {
            if (inputCount < 1) throw new ArgumentOutOfRangeException(nameof(inputCount));
            this._inputCount = inputCount;
        }

        public InputLayerConfig(ILayer layer)
        {
            this._inputCount = (int)layer.output_shape.size;
        }

        public virtual int GetInputCount() => this._inputCount;

        public virtual Tensors ToTensors()
        {
            return new Tensors(KerasApi.keras.Input(new Shape(this._inputCount)));
        }
    }
}
