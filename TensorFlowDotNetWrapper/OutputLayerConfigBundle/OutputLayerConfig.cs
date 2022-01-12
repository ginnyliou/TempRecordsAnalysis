using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Tensorflow;
using Tensorflow.Eager;
using Tensorflow.Keras;

namespace TensorFlowDotNetWrapper.OutputLayerConfigBundle
{
    public class OutputLayerConfig : IOutputLayerConfig
    {
        private class Initializer : IInitializer
        {
            private readonly IReadOnlyCollection<float> _kernels;

            public Initializer(IEnumerable<float> kernels)
            {
                this._kernels = kernels?.ToImmutableArray();
            }

            public Tensor Apply(InitializerArgs args)
            {
                return this._kernels is null
                    ? null
                    : new EagerTensor(this._kernels.ToArray(), args.Shape);
            }
        }

        public enum KnownActivation
        {
            Sigmoid,
            Tanh,
            Softmax
        }
        private readonly int _outputCount;
        private readonly string _activationName;
        private readonly IReadOnlyCollection<float> _kernels;
        private readonly IReadOnlyCollection<float> _bias;

        public OutputLayerConfig(
            int outputCount,
            string activationName = null,
            IEnumerable<float> kernels = null,
            IEnumerable<float> bias = null
        )
        {
            if (outputCount < 1) throw new ArgumentOutOfRangeException(nameof(outputCount));
            this._outputCount = outputCount;
            this._activationName = activationName;
            this._kernels = kernels?.ToImmutableArray();
            this._bias = bias?.ToImmutableArray();
        }

        public OutputLayerConfig(ILayer layer)
        {
            this._outputCount = (int)layer.output_shape.size;
            this._kernels = layer.trainable_weights.ElementAtOrDefault(0)?.numpy().ToArray<float>();
            this._bias = layer.trainable_weights.ElementAtOrDefault(1)?.numpy().ToArray<float>();
        }

        public virtual int GetOutputCount() => this._outputCount;
        public virtual string GetActivationName() => this._activationName;
        public virtual float[] GetKernels() => this._kernels?.ToArray();
        public virtual float[] GetBias() => this._bias?.ToArray();

        public virtual Tensors ToTensors(Tensors input)
        {
            if (input is null) return null;
            Activation activation = KerasApi.keras.activations.Softmax;
            if (Enum.TryParse(this._activationName, out KnownActivation activationRequest))
            {
                switch (activationRequest)
                {
                    case KnownActivation.Sigmoid:
                        activation = KerasApi.keras.activations.Sigmoid;
                        break;
                    case KnownActivation.Tanh:
                        activation = KerasApi.keras.activations.Tanh;
                        break;
                    case KnownActivation.Softmax:
                        activation = KerasApi.keras.activations.Softmax;
                        break;
                    default:
                        activation = KerasApi.keras.activations.Softmax;
                        break;
                }
            }
            Initializer kernelInitializer = this._kernels is null ? null : new Initializer(this._kernels);
            Initializer biasInitializer = this._bias is null ? null : new Initializer(this._bias);
            Tensors result = KerasApi.keras.layers.Dense(
                this._outputCount,
                activation,
                kernelInitializer,
                true,
                biasInitializer
            ).Apply(input);
            return result;
        }
    }
}
