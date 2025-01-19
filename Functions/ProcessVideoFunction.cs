using System;
using System.IO;                   // For StreamReader
using System.Net;
using System.Text.Json;
using System.Text;
using System.Linq;                 // For Any() extension method
using Azure.Storage.Queues;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Linq;
using System.Net.Http.Json;

public class StageInfo
{
    [JsonPropertyName("name")]
    public string Name { get; set; }

    [JsonPropertyName("start_time")]
    public float StartTime { get; set; }

    [JsonPropertyName("end_time")]
    public float EndTime { get; set; }
}

public class ProcessVideoRequest
{
    [JsonPropertyName("exercise")]
    public string Exercise { get; set; }

    [JsonPropertyName("video_url")]
    public string VideoUrl { get; set; }

    [JsonPropertyName("stages")]
    public List<StageInfo> Stages { get; set; }

    [JsonPropertyName("user_id")]
    public string UserId { get; set; }

    [JsonPropertyName("deployment_id")]
    public string DeploymentId { get; set; }
}

public class ProcessVideoFunction
{
    private readonly ILogger _logger;

    public ProcessVideoFunction(ILoggerFactory loggerFactory)
    {
        _logger = loggerFactory.CreateLogger<ProcessVideoFunction>();
    }

    [Function("ProcessVideo")]
    public async Task<HttpResponseData> RunProcessVideo(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = "process_video")] HttpRequestData req)
    {
        _logger.LogInformation("ProcessVideo function triggered.");

        try
        {
            // Read and validate request body
            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            if (string.IsNullOrEmpty(requestBody))
            {
                var badRequestResponse = req.CreateResponse(HttpStatusCode.BadRequest);
                await badRequestResponse.WriteAsJsonAsync(new { error = "Request body is required" });
                return badRequestResponse;
            }

            // Deserialize the request
            var videoRequest = JsonSerializer.Deserialize<ProcessVideoRequest>(requestBody);
            
            // Validate required fields with enhanced stage validation
            if (string.IsNullOrEmpty(videoRequest?.VideoUrl) || 
                string.IsNullOrEmpty(videoRequest?.Exercise) ||
                videoRequest?.Stages == null || 
                !videoRequest.Stages.Any() ||
                videoRequest.Stages.Any(s => string.IsNullOrEmpty(s.Name) || s.EndTime <= s.StartTime))
            {
                var badRequestResponse = req.CreateResponse(HttpStatusCode.BadRequest);
                await badRequestResponse.WriteAsJsonAsync(new { 
                    error = "Invalid request. Ensure all required fields are provided and stage times are valid." 
                });
                return badRequestResponse;
            }

            // Get storage connection string
            var storageConnString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
            if (string.IsNullOrEmpty(storageConnString))
            {
                throw new InvalidOperationException("Storage connection string not configured");
            }

            // Create queue client and ensure queue exists
            var queueClient = new QueueClient(storageConnString, "videoprocess");
            await queueClient.CreateIfNotExistsAsync();

            // Generate a unique processing ID
            var processingId = Guid.NewGuid().ToString();
            
            // Create message payload
            var queueMessage = new
            {
                processingId,
                videoRequest.Exercise,
                videoRequest.VideoUrl,
                videoRequest.Stages,
                videoRequest.UserId,
                videoRequest.DeploymentId,
                Timestamp = DateTimeOffset.UtcNow
            };

            // Convert to Base64 encoded string to handle special characters
            var messageJson = JsonSerializer.Serialize(queueMessage);
            var messageBytes = Encoding.UTF8.GetBytes(messageJson);
            var base64Message = Convert.ToBase64String(messageBytes);

            // Send message to queue
            await queueClient.SendMessageAsync(base64Message);

            // Create success response with processing ID and proper headers
            var response = req.CreateResponse(HttpStatusCode.Accepted);
            response.Headers.Add("Location", $"/api/video_status/{processingId}");
            response.Headers.Add("Retry-After", "60"); // Suggest client to retry after 60 seconds

            await response.WriteAsJsonAsync(new
            {
                processing_id = processingId,
                status = "accepted",
                status_url = $"/api/video_status/{processingId}",
                estimated_completion_time = DateTimeOffset.UtcNow.AddMinutes(5) // Rough estimate
            });

            _logger.LogInformation($"Video processing request queued. Processing ID: {processingId}");
            return response;
        }
        catch (JsonException ex)
        {
            _logger.LogError(ex, "Error parsing request JSON");
            var errorResponse = req.CreateResponse(HttpStatusCode.BadRequest);
            await errorResponse.WriteAsJsonAsync(new { error = "Invalid JSON format" });
            return errorResponse;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing video request");
            var errorResponse = req.CreateResponse(HttpStatusCode.InternalServerError);
            await errorResponse.WriteAsJsonAsync(new { error = "Internal server error" });
            return errorResponse;
        }
    }
}
