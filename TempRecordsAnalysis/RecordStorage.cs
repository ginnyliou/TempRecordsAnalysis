using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TempRecordsAnalysis
{
    public class RecordStorage
    {
        public string StorageBasePath { get; set; }

        private JsonDocument ReadFileAsJsonDocument(string filePath)
        {
            try
            {
                if (!File.Exists(filePath)) return null;
                return JsonDocument.Parse(File.ReadAllText(filePath));
            }
            catch
            {
                return null;
            }
        }

        private void WriteJsonDocumentToFile(string filePath, JsonDocument document)
        {
            new FileInfo(filePath).Directory?.Create();
            File.WriteAllText(filePath, document.RootElement.GetRawText());
        }

        private string GenerateRecordFilePath(TempRecord record)
        {
            string folderPath = Path.Combine(
                this.StorageBasePath,
                $"{record.RecTime:yyyy}"
            );
            string fileName = $"{record.RecTime:yyyy-MM-dd}.json";
            return Path.Combine(
                folderPath,
                fileName
            );
        }

        private string[] FindRecordFilePaths(
            DateTime? fromTimeMark = null,
            DateTime? toTimeMark = null)
        {
            if (!Directory.Exists(this.StorageBasePath)) return null;
            string searchPattern = "*.json";
            return Directory.GetFiles(
                    this.StorageBasePath,
                    searchPattern,
                    SearchOption.AllDirectories
                )
                .Where(ShouldBeSelected)
                .ToArray();

            bool ShouldBeSelected(string filePath)
            {
                if (string.IsNullOrWhiteSpace(filePath)) return false;
                string filePathTimeMark = new string(Path.GetFileName(filePath).Take(10).ToArray());
                if (fromTimeMark.HasValue && string.Compare(
                        filePathTimeMark,
                        $"{fromTimeMark:yyyy-MM-dd}",
                        StringComparison.InvariantCultureIgnoreCase) < 0) return false;
                if (toTimeMark.HasValue && string.Compare(
                        filePathTimeMark,
                        $"{toTimeMark:yyyy-MM-dd}",
                        StringComparison.InvariantCultureIgnoreCase) > 0) return false;
                return true;
            }
        }

        private TempRecord[] ReadRecordsFromFile(string filePath)
        {
            using JsonDocument document = this.ReadFileAsJsonDocument(filePath);
            JsonElement property = new JsonElement();
            bool haveRecords = document?.RootElement.TryGetProperty("records", out property) ?? false;
            return haveRecords && property.ValueKind is JsonValueKind.Array ?
                property.EnumerateArray()
                    .Select(item => new TempRecord(item.EnumerateObject()))
                    .ToArray()
                : null;
        }

        private TempRecord[] ReadRecordsFromFiles(IEnumerable<string> filePaths)
        {
            return filePaths?
                .SelectMany(filepath => this.ReadRecordsFromFile(filepath) ?? Array.Empty<TempRecord>())
                .ToArray();
        }

        private TempRecord[] Cleanse(
            IEnumerable<TempRecord> records,
            DateTime? fromTimeMark = null,
            DateTime? toTimeMark = null)
        {
            return records?
                .Where(r =>
                    (!fromTimeMark.HasValue || r.RecTime >= fromTimeMark)
                    && (!toTimeMark.HasValue || r.RecTime <= toTimeMark)
                )
                .GroupBy(r => $"{r.RecTime:O}")
                .Select(g => g.OrderBy(r => r.RecTime).Last())
                .ToArray();
        }

        public virtual TempRecord[] LoadRecords(
            DateTime? fromTimeMark = null,
            DateTime? toTimeMark = null)
        {
            return this.Cleanse(
                this.ReadRecordsFromFiles(
                    this.FindRecordFilePaths(
                        fromTimeMark,
                        toTimeMark
                    )
                ),
                fromTimeMark,
                toTimeMark
            );
        }

        public virtual void SaveRecords(IEnumerable<TempRecord> records)
        {
            if (records is null) return;
            foreach (IGrouping<string, TempRecord> g in records.GroupBy(this.GenerateRecordFilePath))
            {
                if (g.Key is null) continue;
                List<TempRecord> temp = g.ToList();
                temp.AddRange(this.ReadRecordsFromFile(g.Key) ?? Enumerable.Empty<TempRecord>());
                this.WriteJsonDocumentToFile(
                    g.Key,
                    JsonDocument.Parse(
                        JsonSerializer.Serialize(
                            new Dictionary<string, object>
                            {
                                {
                                    "records",
                                    this.Cleanse(temp).Select(item => item.ToSerializableObject())
                                }
                            }
                        )
                    )
                );
            }
        }
    }
}
