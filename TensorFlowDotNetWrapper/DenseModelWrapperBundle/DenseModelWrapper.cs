using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using TensorFlowDotNetWrapper.DenseLayerConfigBundle;
using TensorFlowDotNetWrapper.InputLayerConfigBundle;
using TensorFlowDotNetWrapper.OutputLayerConfigBundle;

namespace TensorFlowDotNetWrapper.DenseModelWrapperBundle
{
    public class DenseModelWrapper : IDenseModelWrapper
    {
        private readonly IInputLayerConfig _inputLayerConfig;
        private readonly IOutputLayerConfig _outputLayerConfig;
        private readonly IReadOnlyCollection<IDenseLayerConfig> _hiddenLayerConfigs;
        private readonly float? _accuracy;
        private readonly Functional _model;

        public DenseModelWrapper(
            IInputLayerConfig inputLayerConfig,
            IOutputLayerConfig outputLayerConfig,
            IEnumerable<IDenseLayerConfig> hiddenLayerConfigs,
            float? accuracy = null
        )
        {
            if (inputLayerConfig is null) throw new ArgumentNullException(nameof(inputLayerConfig));
            if (outputLayerConfig is null) throw new ArgumentNullException(nameof(outputLayerConfig));
            this._inputLayerConfig = inputLayerConfig;
            this._outputLayerConfig = outputLayerConfig;
            this._hiddenLayerConfigs = hiddenLayerConfigs?.ToImmutableArray();
            this._accuracy = accuracy;
            // Create the dense model.
            this._model = this.CreateDenseModel(
                this._inputLayerConfig,
                this._outputLayerConfig,
                this._hiddenLayerConfigs
            );
        }

        private Functional CreateDenseModel(
            IInputLayerConfig inputLayerConfig,
            IOutputLayerConfig outputLayerConfig,
            IEnumerable<IDenseLayerConfig> hiddenLayerConfigs
        )
        {
            List<Tensors> layers = new List<Tensors>();
            layers.add(inputLayerConfig.ToTensors());
            foreach (IDenseLayerConfig hiddenLayerConfig in hiddenLayerConfigs ?? Enumerable.Empty<IDenseLayerConfig>())
            {
                layers.add(hiddenLayerConfig.ToTensors(layers.Last()));
            }
            layers.add(outputLayerConfig.ToTensors(layers.Last()));
            Functional model = KerasApi.keras.Model(
                layers.First(),
                layers.Last()
            );
            model.compile(
                KerasApi.keras.optimizers.SGD(0.001f),
                KerasApi.keras.losses.CategoricalCrossentropy(from_logits: true),
                new[] { "accuracy" }
            );
            return model;
        }

        public virtual IInputLayerConfig GetInputLayerConfig() => this._inputLayerConfig;
        public virtual IOutputLayerConfig GetOutputLayerConfig() => this._outputLayerConfig;
        public virtual IDenseLayerConfig[] GetHiddenLayerConfigs() => this._hiddenLayerConfigs?.ToArray();
        public virtual float? GetAccuracy() => this._accuracy;

        public virtual IDenseModelWrapper Train(float[,] data, float[,] result, int epochs = 1, float validationSplit = 0.25f)
        {
            // Create new model based on current layers.
            Functional tempModel = this.CreateDenseModel(
                this._inputLayerConfig,
                this._outputLayerConfig,
                this._hiddenLayerConfigs
            );
            // Fit new model.
            tempModel.fit(
                np.array(data),
                np.array(result),
                epochs: epochs < 1 ? 1 : epochs,
                validation_split: validationSplit,
                verbose: 1
            );
            // Extract from the new model.
            InputLayerConfig inputLayerConfig = null;
            OutputLayerConfig outputLayerConfig = null;
            List<IDenseLayerConfig> hiddenLayerConfigs = new List<IDenseLayerConfig>();
            foreach (ILayer layer in tempModel.Layers.Where(l => l is InputLayer or Dense))
            {
                if (layer is InputLayer)
                {
                    inputLayerConfig = new InputLayerConfig(layer);
                }
                else
                {
                    if (hiddenLayerConfigs.Count != (this._hiddenLayerConfigs?.Count ?? 0))
                    {
                        IDenseLayerConfig tempLayer = this._hiddenLayerConfigs?.ElementAtOrDefault(hiddenLayerConfigs.Count);
                        hiddenLayerConfigs.add(new DenseLayerConfig(layer, tempLayer?.GetActivationName(), tempLayer?.GetDropoutRate()));
                    }
                    outputLayerConfig = new OutputLayerConfig(layer);
                }
            }
            float? accuracy = tempModel
                .metrics
                .FirstOrDefault(m => m.Name == "accuracy")?
                .result()
                .numpy()
                .ToArray<float>().FirstOrDefault();
            // Return new model wrapper.
            return new DenseModelWrapper(
                inputLayerConfig,
                outputLayerConfig,
                hiddenLayerConfigs,
                accuracy
            );
        }

        public virtual float[] Predict(float[] data)
        {
            if (data is null) return null;
            float[,] temp = new float[1, data.Length];
            for (int i = 0; i < data.Length; i++) temp[0, i] = data[i];
            return this._model?
                .predict(np.array(temp))
                .FirstOrDefault()?
                .numpy()
                .ToArray<float>();
        }
    }
}
