using System;
using System.Threading.Tasks;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using System.Net;
using Microsoft.Extensions.Logging;

namespace backend.Functions
{
    public class VideoStatusFunction
    {
        private readonly ILogger<VideoStatusFunction> _logger;

        public VideoStatusFunction(ILogger<VideoStatusFunction> logger)
        {
            _logger = logger;
        }

        [Function("GetVideoStatus")]
        public async Task<HttpResponseData> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", Route = "status/{processingId}")] HttpRequestData req,
            [TableInput("VideoProcessing", "{processingId}", "status")] VideoStatus statusEntity,
            string processingId)
        {
            _logger.LogInformation($"Getting status for processing ID: {processingId}");

            if (statusEntity == null)
            {
                var notFoundResponse = req.CreateResponse(HttpStatusCode.NotFound);
                await notFoundResponse.WriteAsJsonAsync(new { error = $"No status found for processing ID: {processingId}" });
                return notFoundResponse;
            }

            var response = req.CreateResponse(HttpStatusCode.OK);
            await response.WriteAsJsonAsync(new
            {
                processingId = processingId,
                status = statusEntity.Status,
                message = statusEntity.Message,
                timestamp = statusEntity.Timestamp
            });

            return response;
        }
    }
}

public class VideoStatus
{
    public string PartitionKey { get; set; }
    public string RowKey { get; set; }
    public string Status { get; set; }
    public string Message { get; set; }
    public DateTime Timestamp { get; set; }
} 