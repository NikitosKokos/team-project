using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Net.Http;
using System.Net.Http.Json;
using Microsoft.Azure.Functions.Worker;
using Azure.Data.Tables;
using Azure.Storage.Blobs;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Linq;
using System.Net.Http.Json;

public class VideoProcessMessage
{
    public string ProcessingId { get; set; }
    public string Exercise { get; set; }
    public string VideoUrl { get; set; }
    public List<StageInfo> Stages { get; set; }
    public string UserId { get; set; }
    public string DeploymentId { get; set; }
    public DateTimeOffset Timestamp { get; set; }
}

public class VideoAnalysisResult
{
    public string ProcessingId { get; set; }
    public Dictionary<string, object> StageAnalysis { get; set; }
    public List<string> Warnings { get; set; }
    public Dictionary<string, double> Metrics { get; set; }
}

public class VideoProcessQueueTrigger
{
    private readonly ILogger _logger;
    private readonly HttpClient _httpClient;
    private readonly string _pythonServiceUrl;

    public VideoProcessQueueTrigger(ILoggerFactory loggerFactory, IHttpClientFactory httpClientFactory)
    {
        _logger = loggerFactory.CreateLogger<VideoProcessQueueTrigger>();
        _httpClient = httpClientFactory.CreateClient();
        _httpClient.Timeout = TimeSpan.FromMinutes(5);
        _pythonServiceUrl = Environment.GetEnvironmentVariable("PythonServiceUrl") 
            ?? throw new InvalidOperationException("PythonServiceUrl not configured");
    }

    [Function("VideoProcessQueueTrigger")]
    public async Task ProcessVideoQueue(
        [QueueTrigger("videoprocess", Connection = "AzureWebJobsStorage")] string base64Message)
    {
        try
        {
            // Decode the base64 message
            var messageBytes = Convert.FromBase64String(base64Message);
            var messageJson = Encoding.UTF8.GetString(messageBytes);
            var message = JsonSerializer.Deserialize<VideoProcessMessage>(messageJson);

            _logger.LogInformation($"Processing video request {message.ProcessingId}");

            // Update status in Table Storage
            await UpdateProcessingStatus(message.ProcessingId, "processing", "Video analysis started");

            // Prepare request for Python service
            var pythonRequest = new
            {
                message.VideoUrl,
                message.Exercise,
                message.Stages,
                message.DeploymentId
            };

            // Call Python service
            var response = await _httpClient.PostAsJsonAsync($"{_pythonServiceUrl}/analyze", pythonRequest);
            
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                throw new Exception($"Python service returned {response.StatusCode}: {errorContent}");
            }

            // Use the typed model for analysis results
            var analysisResult = await response.Content.ReadFromJsonAsync<VideoAnalysisResult>();

            // Store results in Blob Storage
            await StoreResults(message.ProcessingId, analysisResult);

            // Update status to completed
            await UpdateProcessingStatus(message.ProcessingId, "completed", "Analysis completed successfully");

            _logger.LogInformation($"Video processing completed for {message.ProcessingId}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing video queue message");
            
            try
            {
                // Extract processing ID from the message even if it failed to deserialize
                string processingId = "unknown";
                if (!string.IsNullOrEmpty(base64Message))
                {
                    var messageBytes = Convert.FromBase64String(base64Message);
                    var messageJson = Encoding.UTF8.GetString(messageBytes);
                    if (JsonDocument.Parse(messageJson).RootElement.TryGetProperty("processingId", out var idElement))
                    {
                        processingId = idElement.GetString();
                    }
                }

                // Update status to failed
                await UpdateProcessingStatus(processingId, "failed", ex.Message);
            }
            catch (Exception statusEx)
            {
                _logger.LogError(statusEx, "Failed to update error status");
            }

            throw; // Rethrow to trigger Azure Functions retry policy
        }
    }

    private async Task UpdateProcessingStatus(string processingId, string status, string message)
    {
        var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
        var tableClient = new TableClient(connectionString, "VideoProcessing");
        await tableClient.CreateIfNotExistsAsync();

        var entity = new TableEntity(processingId, "status")
        {
            { "Status", status },
            { "Message", message },
            { "LastUpdated", DateTimeOffset.UtcNow }
        };

        await tableClient.UpsertEntityAsync(entity);
    }

    private async Task StoreResults(string processingId, object results)
    {
        var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
        var blobClient = new BlobContainerClient(connectionString, "results");
        await blobClient.CreateIfNotExistsAsync();

        var blob = blobClient.GetBlobClient($"{processingId}.json");
        var content = JsonSerializer.Serialize(results);
        var bytes = Encoding.UTF8.GetBytes(content);
        using var stream = new MemoryStream(bytes);
        await blob.UploadAsync(stream, overwrite: true);
    }
}
