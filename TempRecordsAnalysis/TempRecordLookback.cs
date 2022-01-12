using System.Collections.Generic;
using System.Linq;

namespace TempRecordsAnalysis
{
    public class TempRecordLookback
    {
        public TempRecord TargetRecord { get; set; }
        public List<TempRecord> PreviousRecords { get; set; }

        public (float[] Inputs, float[] Outputs) ToInputsOutputs()
        {
            float[] inputs = this.PreviousRecords
                .OrderByDescending(pr => pr.RecTime)
                .SelectMany(pr =>
                {
                    List<float> result = new List<float>();
                    result.Add(pr.TempChange > 0 ? 1f : 0f);
                    result.Add(pr.TempChange < 0 ? 1f : 0f);
                    return result;
                }
                ).ToArray();
            float[] outputs = new float[]
            {
                TargetRecord.TempChange < 0 ? 1f : 0f,
                TargetRecord.TempChange == 0 ? 1f : 0f,
                TargetRecord.TempChange > 0 ? 1f : 0f
            };
            return (inputs, outputs);
        }
    }
}
