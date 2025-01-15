using Azure.Storage.Blobs;
using System;
using System.IO;
using System.Threading.Tasks;

namespace backend.Services
{
    public class BlobService
    {
        private readonly string _blobConnectionString;
        private readonly string _containerName;

        public BlobService(string blobConnectionString, string containerName)
        {
            _blobConnectionString = blobConnectionString;
            _containerName = containerName;
        }

        public async Task<string> UploadVideoAsync(string fileName, Stream fileStream)
        {
            try
            {
                var blobServiceClient = new BlobServiceClient(_blobConnectionString);
                var blobContainerClient = blobServiceClient.GetBlobContainerClient(_containerName);

                // Ensure the container exists
                await blobContainerClient.CreateIfNotExistsAsync();

                // Generate a unique filename to avoid overwriting
                var processingId = Guid.NewGuid().ToString();
                var uniquePath = $"videos/{processingId}/{fileName}";
                var blobClient = blobContainerClient.GetBlobClient(uniquePath);


                // Upload the file
                await blobClient.UploadAsync(fileStream);

                // Return the URL of the uploaded file
                return blobClient.Uri.ToString();
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to upload video to Blob Storage: {ex.Message}");
            }
        }
    }
}
