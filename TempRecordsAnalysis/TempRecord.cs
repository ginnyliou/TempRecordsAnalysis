using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TempRecordsAnalysis
{
    public class TempRecord
    {
        public DateTime RecTime { get; set; }
        public decimal Start { get; set; }
        public decimal End { get; set; }
        public decimal High { get; set; }
        public decimal Low { get; set; }
        public int TempChange => Math.Sign(End - Start);

        private static readonly JsonSerializerOptions DeserializeOption = new JsonSerializerOptions
        {
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals |
                             JsonNumberHandling.AllowReadingFromString
        };

        public TempRecord() { }

        public TempRecord(JsonElement.ObjectEnumerator enumerator)
        {
            foreach (JsonProperty property in enumerator)
            {
                string name = property.Name;
                JsonElement value = property.Value;
                if (name == "recTime") this.RecTime = Deserialize<DateTime>(value);
                if (name == "start") this.Start = Deserialize<decimal>(value);
                if (name == "end") this.End = Deserialize<decimal>(value);
                if (name == "high") this.High = Deserialize<decimal>(value);
                if (name == "low") this.Low = Deserialize<decimal>(value);
            }

            T Deserialize<T>(JsonElement element)
            {
                return JsonSerializer.Deserialize(element.GetRawText(), typeof(T), DeserializeOption) is T result
                    ? result
                    : default;
            }
        }

        public IDictionary<string, object> ToSerializableObject()
        {
            return new Dictionary<string, object>
            {
                { "recTime", this.RecTime },
                { "start", this.Start },
                { "end", this.End },
                { "high", this.High },
                { "low", this.Low }
            };
        }

        public virtual string ToJsonString(JsonSerializerOptions options = null)
        {
            return JsonSerializer.Serialize(this.ToSerializableObject(), options);
        }
    }
}
