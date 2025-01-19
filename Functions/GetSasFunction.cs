using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using System;
using System.IO;
using System.Net;
using System.Text.Json;
using System.Threading.Tasks;
using Azure.Storage;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Azure.Storage.Blobs.Specialized;
using Azure.Storage.Sas;
using Microsoft.Extensions.Logging;
using System.Linq;
using System.Net.Http.Json;

namespace backend.Functions
{
    public class GetSasFunction
    {
        private readonly ILogger<GetSasFunction> _logger;

        public GetSasFunction(ILogger<GetSasFunction> logger)
        {
            _logger = logger;
        }

        [Function("GetSasToken")]
        public async Task<HttpResponseData> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get")] HttpRequestData req)
        {
            _logger.LogInformation("GetSas function triggered.");

            // Parse query parameters
            var query = System.Web.HttpUtility.ParseQueryString(req.Url.Query);
            var filename = query["filename"];
            if (string.IsNullOrEmpty(filename))
            {
                var response = req.CreateResponse(HttpStatusCode.BadRequest);
                await response.WriteAsJsonAsync(new { error = "Filename parameter is required" });
                return response;
            }

            try
            {
                // Get connection string from configuration
                var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage");
                if (string.IsNullOrEmpty(connectionString))
                {
                    throw new InvalidOperationException("Storage connection string not configured");
                }

                // Create BlobServiceClient
                var blobServiceClient = new BlobServiceClient(connectionString);
                var containerClient = blobServiceClient.GetBlobContainerClient("uploads");

                // Ensure container exists
                await containerClient.CreateIfNotExistsAsync(PublicAccessType.None);

                // Get blob client for the specific file
                var blobClient = containerClient.GetBlobClient(filename);

                // Generate SAS token with read and write permissions
                var sasBuilder = new BlobSasBuilder
                {
                    BlobContainerName = containerClient.Name,
                    BlobName = blobClient.Name,
                    Resource = "b",
                    StartsOn = DateTimeOffset.UtcNow,
                    ExpiresOn = DateTimeOffset.UtcNow.AddMinutes(30), // Extended to 30 minutes
                };
                sasBuilder.SetPermissions(BlobSasPermissions.Write | BlobSasPermissions.Read);

                // Generate the SAS token
                // Fallback approach if GenerateSasUri fails
                string sasUrl;
                try 
                {
                    var sasUri = blobClient.GenerateSasUri(sasBuilder);
                    sasUrl = sasUri.ToString();
                }
                catch (InvalidOperationException)
                {
                    // Fallback to manual SAS generation
                    var storageSharedKeyCredential = new StorageSharedKeyCredential(
                        blobServiceClient.AccountName,
                        Environment.GetEnvironmentVariable("StorageAccountKey")
                    );
                    
                    var sasQueryParameters = sasBuilder.ToSasQueryParameters(storageSharedKeyCredential);
                    sasUrl = $"{blobClient.Uri}?{sasQueryParameters}";
                }

                // Create success response
                var successResponse = req.CreateResponse(HttpStatusCode.OK);
                await successResponse.WriteAsJsonAsync(new { sas_url = sasUrl });
                return successResponse;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating SAS token");
                var errorResponse = req.CreateResponse(HttpStatusCode.InternalServerError);
                await errorResponse.WriteAsJsonAsync(new { error = "Failed to generate SAS token" });
                return errorResponse;
            }
        }
    }
}
