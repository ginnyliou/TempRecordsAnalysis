using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Tensorflow;
using Tensorflow.Eager;
using Tensorflow.Keras;

namespace TensorFlowDotNetWrapper.DenseLayerConfigBundle
{
    public class DenseLayerConfig : IDenseLayerConfig
    {
        private class Initializer : IInitializer
        {
            private readonly IReadOnlyCollection<float> _kernels;

            public Initializer(IEnumerable<float> kernels)
            {
                this._kernels = kernels.ToImmutableArray();
            }

            public Tensor Apply(InitializerArgs args)
            {
                return new EagerTensor(this._kernels.ToArray(), args.Shape);
            }
        }

        public enum KnownActivation
        {
            None,
            Sigmoid,
            Tanh,
            ReLU,
            Softmax
        }
        private readonly int _neuronCount;
        private readonly string _activationName;
        private readonly float? _dropoutRate;
        private readonly IReadOnlyCollection<float> _kernels;
        private readonly IReadOnlyCollection<float> _bias;

        public DenseLayerConfig(
            int neuronCount,
            string activationName = null,
            float? dropoutRate = null,
            IEnumerable<float> kernels = null,
            IEnumerable<float> bias = null
        )
        {
            this._neuronCount = neuronCount;
            this._activationName = activationName;
            this._dropoutRate = dropoutRate;
            this._kernels = kernels?.ToImmutableArray();
            this._bias = bias?.ToImmutableArray();
        }

        public DenseLayerConfig(
            ILayer layer,
            string activationName = null,
            float? dropoutRate = null
        )
        {
            this._neuronCount = (int)layer.output_shape.size;
            this._activationName = activationName;
            this._dropoutRate = dropoutRate;
            this._kernels = layer.trainable_weights.ElementAtOrDefault(0)?.numpy().ToArray<float>();
            this._bias = layer.trainable_weights.ElementAtOrDefault(1)?.numpy().ToArray<float>();
        }

        public virtual int GetNeuronCount() => this._neuronCount;
        public virtual string GetActivationName() => this._activationName;
        public virtual float? GetDropoutRate() => this._dropoutRate;
        public virtual float[] GetKernels() => this._kernels?.ToArray();
        public virtual float[] GetBias() => this._bias?.ToArray();

        public virtual Tensors ToTensors(Tensors input)
        {
            if (input is null) return null;
            Activation activation = KerasApi.keras.activations.Linear;
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
                    case KnownActivation.ReLU:
                        activation = KerasApi.keras.activations.Relu;
                        break;
                    case KnownActivation.Softmax:
                        activation = KerasApi.keras.activations.Softmax;
                        break;
                    default:
                        activation = KerasApi.keras.activations.Linear;
                        break;
                }
            }
            Initializer kernelInitializer = this._kernels is null ? null : new Initializer(this._kernels);
            Initializer biasInitializer = this._bias is null ? null : new Initializer(this._bias);
            Tensors result = KerasApi.keras.layers.Dense(
                this._neuronCount,
                activation,
                kernelInitializer,
                true,
                biasInitializer
            ).Apply(input);
            if (this._dropoutRate.HasValue) result = KerasApi.keras.layers.Dropout(this._dropoutRate.Value).Apply(result);
            return result;
        }
    }
}
